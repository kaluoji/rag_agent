# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import feedparser
import tiktoken

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Any
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import deque
from contextlib import asynccontextmanager
from report_agent import report_agent, ReportDeps  # Importa el agente de Reports


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100000
MAX_CHUNKS_RETURNED = 4


load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4-turbo-preview')
model = OpenAIModel(llm)


logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are a powerful expert in data protection regulation – a Python AI agent with access to comprehensive and up-to-date documentation on data protection standards and privacy laws.

Your primary responsibility is to answer regulatory queries with precise, detailed, and well-structured information strictly related to data protection regulation. When providing responses, include all relevant key details and context, and when applicable, enumerate steps or break down the information into clear sections. In cases where the available documentation does not cover every specific detail, use your extensive training and the provided documentation to infer and deliver the best possible, comprehensive answer. Do not ask for additional details or include apologies for missing information—instead, synthesize the most accurate and complete answer based on what is available.

Tools Available:
retrieve_relevant_documentation: Use this tool to retrieve and summarize the most relevant documentation chunks from the comprehensive data protection regulation documentation. Provide a concise summary of the key points.
retrieve_detailed_information: Use this tool to fetch a more granular and in-depth view of the documentation when further clarification or detailed insights are required.
cross_reference_information: Use this tool to link the regulatory query with related content stored in the database, ensuring that all relevant regulations are connected and contextualized.
generate_report: Use this tool to create detailed, structured reports on data protection compliance, audits, or risk assessments. This tool should compile key insights and regulatory data into comprehensive reports.

Workflow:
For general overviews or summaries:
- Invoke retrieve_relevant_documentation to extract and summarize key points.
For queries requiring in-depth details or technical specifications:
- Invoke retrieve_detailed_information to provide a comprehensive, step-by-step explanation.
For queries involving cross-references between regulatory aspects:
- Invoke cross_reference_information to connect and contextualize related regulatory content.
For queries that require the generation of detailed reports:
- Invoke generate_report only when the user's query includes explicit instructions such as "Generate the document", "Generate a report", or similar phrases. Do not call generate_report unless these explicit instructions are present.

Important Notes:
Respond strictly to queries about data protection regulation, privacy laws, and related compliance matters.
Your responses must be detailed, well-organized, and structured. Use numbered lists or bullet points where applicable to clearly present complex processes.
Every time you invoke a tool using a 'tool_call', ensure that the next message in the conversation is a 'tool_return' message that contains the complete response from that tool. Do not omit the tool response under any circumstances, as this is essential for maintaining the integrity of the conversation flow.
If the documentation is incomplete or ambiguous, do not include phrases like "I cannot retrieve information" or "please provide more details." Instead, synthesize the best answer possible using all available resources.
Clearly indicate if the response is a summary, a detailed explanation, a cross-referenced answer, or a generated report.
Use only the necessary tool(s) based on the user's specific query, minimizing unnecessary API calls.
"""


ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2
)

# -------------------- Herramientas del agente --------------------

async def debug_run_agent(user_query: str):
    # Imprime o loguea la consulta que estás a punto de enviar al agente
    logger.debug("Voy a llamar al agente con la query: %s", user_query)

    # Llamada real al agente
    response = await ai_expert.run(user_query=user_query)
    usage_info = response.get("usage")
    logger.info("Uso de tokens en la consulta: %s", usage_info)

    return response

def count_tokens(text: str) -> int:
    """
    Cuenta la cantidad de tokens en un texto utilizando la codificación propia del modelo.
    """
    encoding = tiktoken.encoding_for_model(llm)
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    """
    Trunca el texto para que contenga como máximo max_tokens tokens.
    """
    encoding = tiktoken.encoding_for_model(llm)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text: {text[:30]}... {embedding[:5]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No relevant documentation found."
        
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': MAX_CHUNKS_RETURNED,
                'filter': {} # Filtro vacío
            }
        ).execute()
        
        if not result.data:
            logger.info("No relevant documentation found for the query.")
            return "No relevant documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            logger.debug(f"Retrieved document: {doc['title']}")

        logger.info(f"Returning {len(formatted_chunks)} relevant documentation chunks.")    

        combined_text = "\n\n---\n\n".join(formatted_chunks)
        total_tokens = count_tokens(combined_text)
        logger.debug(f"Total tokens in combined documentation: {total_tokens}")

        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info("El contenido combinado excede el límite de tokens. Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            for chunk in formatted_chunks:
                chunk_tokens = count_tokens(chunk)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens)
                    truncated_chunks.append(truncated_chunk)
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
            combined_text = "\n\n---\n\n".join(truncated_chunks)
            logger.debug(f"After truncation, total tokens: {count_tokens(combined_text)}")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

# ===================== FIN DE LAS HERRAMIENTAS DEL AGENTE =====================

@ai_expert.tool
async def retrieve_detailed_information(ctx: RunContext[AIDeps], user_query: str) -> str:
    """
    Recupera información de la documentación con mayor nivel de detalle basada en la consulta.
    Esta herramienta obtiene más fragmentos para proporcionar datos más granulares.
    """
    try:
        # Obtener el embedding de la consulta
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("El vector embedding resultó ser cero. No se puede proceder con la búsqueda detallada.")
            return "Lo siento, no pude procesar tu consulta para obtener detalles."

        # Se define un mayor número de fragmentos para información detallada
        MAX_DETAILED_CHUNKS = 3
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': MAX_DETAILED_CHUNKS,
                'filter': {}  # Puedes agregar filtros específicos si tu DB lo permite
            }
        ).execute()

        if not result.data or len(result.data) == 0:
            logger.info("No se encontró documentación detallada para la consulta.")
            return "No se encontró información detallada relevante para tu consulta."

        detailed_chunks = []
        for doc in result.data:
            # Se asume que 'content' contiene la información completa o detallada
            chunk_text = f"""
## {doc['title']}

{doc['content']}
"""
            detailed_chunks.append(chunk_text)
            logger.debug(f"Detalle recuperado: {doc['title']}")

        detailed_text = "\n\n---\n\n".join(detailed_chunks)
        total_tokens = count_tokens(detailed_text)
        logger.debug(f"Total de tokens en la información detallada: {total_tokens}")

        # Verificar que el contenido no exceda el límite de tokens
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info("El contenido detallado excede el límite de tokens. Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            for chunk in detailed_chunks:
                chunk_tokens = count_tokens(chunk)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens)
                    truncated_chunks.append(truncated_chunk)
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
            detailed_text = "\n\n---\n\n".join(truncated_chunks)
            logger.debug(f"Tokens después del truncamiento: {count_tokens(detailed_text)}")

        return detailed_text

    except Exception as e:
        logger.error(f"Error recuperando información detallada: {str(e)}")
        return f"Lo siento, ocurrió un error al recuperar la información detallada: {str(e)}"


@ai_expert.tool
async def generate_compliance_report(ctx: RunContext[AIDeps], compliance_output: str, template: str = None) -> str:
    # Verifica que se proporcione un output válido
    if not compliance_output or compliance_output.strip() == "":
        return "Por favor, proporciona el output completo del agente de Compliance para generar el informe."
    
    # Combina el output de Compliance y la plantilla (si existe)
    query = compliance_output
    if template:
        query += f"\n\nUtiliza la siguiente plantilla para el reporte:\n{template}"
    
    # Llama al agente de Reports sin el parámetro 'tool_name'
    report_result = await report_agent.run(
        query,
        deps=ReportDeps(openai_client=ctx.deps.openai_client),
        usage=ctx.usage
    )
    return report_result.data


