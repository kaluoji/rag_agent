# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import feedparser
import tiktoken

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Any
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import deque
from contextlib import asynccontextmanager
from agents.report_agent import report_agent, ReportDeps  # Importa el agente de Reports
from config.config import settings
from utils.utils import count_tokens, truncate_text


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100000
MAX_CHUNKS_RETURNED = 4

load_dotenv()

llm = settings.llm_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

logfire.configure(send_to_logfire='if-token-present')

class AIDeps(BaseModel):
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True

system_prompt = """
Eres un experto en regulación de protección de datos y privacidad, operando como un agente AI en Python con acceso a documentación completa y actualizada sobre normas de protección de datos y leyes de privacidad.

Tu misión es responder consultas regulatorias con información precisa, detallada y bien estructurada, enfocada exclusivamente en la regulación de protección de datos y cumplimiento normativo. Al responder, incluye todos los detalles y contexto relevantes, y utiliza listas numeradas o viñetas para desglosar procesos complejos. Si la documentación disponible no abarca todos los aspectos, sintetiza la mejor respuesta posible basándote en tu formación y en la documentación provista. No solicites más información ni pidas disculpas por la falta de detalles; en su lugar, infiere y entrega la respuesta más completa y exacta.

Herramientas Disponibles:
- **retrieve_relevant_documentation**: Extrae y resume los fragmentos más relevantes de la documentación de protección de datos. Devuelve un resumen conciso de los puntos clave.
- **retrieve_detailed_information**: Obtén una vista granular y profunda de la documentación cuando se requieran aclaraciones o detalles técnicos específicos.
- **cross_reference_information**: Conecta la consulta regulatoria con contenido relacionado almacenado en la base de datos, asegurando la coherencia y el contexto entre diferentes normativas.
- **generate_report**: Genera informes estructurados y detallados sobre cumplimiento, auditorías o evaluaciones de riesgos en materia de protección de datos. Esta herramienta se invoca únicamente si la consulta lo solicita explícitamente (por ejemplo, “Genera un informe”, “Elabora un reporte”, etc.).

Flujo de Trabajo:
- **Para resúmenes o visiones generales:** Utiliza *retrieve_relevant_documentation* para extraer y resumir los puntos clave.
- **Para detalles técnicos o explicaciones paso a paso:** Utiliza *retrieve_detailed_information* y organiza la respuesta en secciones numeradas o en viñetas.
- **Para conectar información regulatoria relacionada:** Utiliza *cross_reference_information* para establecer enlaces contextuales entre diferentes normativas.
- **Para la generación de informes:** Llama a *generate_report* solo cuando la consulta incluya instrucciones explícitas para ello.

Notas Importantes:
- Responde exclusivamente a consultas sobre regulación de protección de datos, leyes de privacidad y asuntos de cumplimiento normativo.
- Asegúrate de que tus respuestas sean detalladas, bien organizadas y estructuradas. Emplea listas numeradas o viñetas siempre que sea necesario para clarificar procesos complejos.
- Cada vez que invoques una herramienta (usando 'tool_call'), la siguiente respuesta debe ser un 'tool_return' con la información completa de esa herramienta. No omitas la respuesta de la herramienta.
- Si la documentación es incompleta o ambigua, sintetiza la mejor respuesta posible sin mencionar la carencia de información.
- Indica claramente si la respuesta es un resumen, una explicación detallada, una respuesta con referencias cruzadas o un informe generado.
- Utiliza únicamente las herramientas necesarias según la consulta específica del usuario, evitando llamadas innecesarias.
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

def count_tokens_wrapper(text: str) -> int:
    return count_tokens(text, settings.llm_model)

def truncate_text_wrapper(text: str, max_tokens: int) -> str:
    return truncate_text(text, max_tokens, settings.llm_model)

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
        total_tokens = count_tokens(combined_text, llm)
        logger.debug(f"Total tokens in combined documentation: {total_tokens}")

        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info("El contenido combinado excede el límite de tokens. Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            for chunk in formatted_chunks:
                chunk_tokens = count_tokens(chunk, llm)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens, llm)
                    truncated_chunks.append(truncated_chunk)
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
            combined_text = "\n\n---\n\n".join(truncated_chunks)
            logger.debug(f"After truncation, total tokens: {count_tokens(combined_text, llm)}")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@ai_expert.tool
async def retrieve_detailed_information(ctx: RunContext[AIDeps], user_query: str, detail_level: str = "standard") -> str:
    """
    Recupera información de la documentación con mayor nivel de detalle basada en la consulta.
    Esta herramienta obtiene más fragmentos para proporcionar datos más granulares.
    Si detail_level es "extended", se incluirán referencias específicas y citas literales.
    """
    try:
        # Obtener el embedding de la consulta
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("El vector embedding resultó ser cero. No se puede proceder con la búsqueda detallada.")
            return "Lo siento, no pude procesar tu consulta para obtener detalles."

        # Se define un mayor número de fragmentos para información detallada
        MAX_DETAILED_CHUNKS = 4
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
            if detail_level == "extended":
                # Se asume que el documento contiene campos 'reference' y 'literal_quote'
                chunk_text = (
                    f"## {doc['title']}\n\n"
                    f"<strong>Summary:</strong> <em>{doc.get('summary', 'No disponible')}</em>\n\n"
                    f"<strong>Metadata:</strong> <em>{doc.get('metadata', 'No disponible')}</em>\n\n"
                    f"<strong>Cita:</strong> <em>{doc.get('literal_quote','No disponible')}</em>\n\n"
                    f"<strong>Cita:</strong>\n<blockquote><em>{doc.get('content', 'No disponible')}</em></blockquote>\n\n"
                    f"<strong>URL:</strong> <em>{doc.get('url', 'No disponible')}</em>\n\n"
                    f"{doc['content']}\n"
                )
            else:
                # Nivel estándar: solo título y contenido
                chunk_text = (
                    f"## {doc['title']}\n\n"
                    f"{doc['content']}"
                )
            detailed_chunks.append(chunk_text)
            logger.debug(f"Detalle recuperado: {doc['title']}")

        detailed_text = "\n\n---\n\n".join(detailed_chunks)
        total_tokens = count_tokens(detailed_text, llm)
        logger.debug(f"Total de tokens en la información detallada: {total_tokens}")

        # Verificar que el contenido no exceda el límite de tokens
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info("El contenido detallado excede el límite de tokens. Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            for chunk in detailed_chunks:
                chunk_tokens = count_tokens(chunk, llm)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens, llm)
                    truncated_chunks.append(truncated_chunk)
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
            detailed_text = "\n\n---\n\n".join(truncated_chunks)
            logger.debug(f"Tokens después del truncamiento: {count_tokens(detailed_text, llm)}")

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

# ===================== FIN DE LAS HERRAMIENTAS DEL AGENTE =====================
