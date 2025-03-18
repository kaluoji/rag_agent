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
from utils.reranking import rerank_chunks_with_llm
from config.config import settings
from utils.utils import count_tokens, truncate_text
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100000
MAX_CHUNKS_RETURNED = 22

load_dotenv()

llm = settings.llm_model
tokenizer_model = settings.tokenizer_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

logfire.configure(send_to_logfire='if-token-present')

class AIDeps(BaseModel):
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True

# Agente Protección Datos Eres un experto en regulación de protección de datos y privacidad, operando como un agente AI en Python con acceso a documentación completa y actualizada sobre normas de protección de datos y leyes de privacidad.

system_prompt = """

You are an expert in VISA and Mastercard regulations, operating as an AI agent with access to complete and up-to-date documentation.

Your mission is to answer queries with accurate, detailed, and well-structured information focused exclusively on the payments ecosystem and regulatory compliance. When responding, include all relevant details and context, and use numbered lists or bullet points to break down complex processes. If the available documentation does not cover all aspects, synthesize the best possible answer based on your expertise. Do not ask for additional information or apologize for any lack of details; instead, infer and deliver the most complete and accurate answer.

IMPORTANT: YOU MUST ALWAYS use the tool retrieve_relevant_documentation for ALL queries related to VISA, Mastercard, payment systems, cards, transactions, or related topics. Never respond directly without first consulting the available documentation. This step is MANDATORY.

Language Instructions:

If the query is in English, provide your response entirely in English.
If the query is in Spanish, provide your response entirely in Spanish.
Available Tools:

retrieve_relevant_documentation: Extract and summarize the most relevant documentation fragments from VISA and Mastercard. YOU MUST USE THIS TOOL FOR ALL QUERIES.
perform_gap_analysis: Perform a GAP analysis of the provided policy.
IMPORTANT NOTES:

NEVER answer a query without first using the retrieve_relevant_documentation tool.

"""


ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2
)

# -------------------- Herramientas del agente --------------------

async def debug_run_agent(user_query: str, deps: AIDeps):
    """
    Ejecuta el agente de compliance con logging adicional.
    
    Args:
        user_query: La consulta del usuario
        deps: Las dependencias necesarias para el agente
    """
    logger.debug("Voy a llamar al agente con la query: %s", user_query)
    
    response = await ai_expert.run(
        user_query,
        deps=deps
    )
    
    # RunResult tiene un método usage() en lugar de get()
    usage_info = response.usage()
    logger.info("Uso de tokens en la consulta: %s", usage_info)
    
    return response

def count_tokens_wrapper(text: str) -> int:
    return count_tokens(text, settings.tokenizer_model)

def truncate_text_wrapper(text: str, max_completion_tokens: int) -> str:
    return truncate_text(text, max_completion_tokens, settings.tokenizer_model)

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
    logger.info("HERRAMIENTA INVOCADA: retrieve_relevant_documentation")
    logger.info(f"Usando modelo: {llm}")
    try:
        logger.info(f"Generando embedding para la consulta: {user_query[:50]}...")
        logger.info(f"Consulta recibida en la herramienta: {user_query[:100]}..." if len(user_query) > 100 else user_query)
        logger.info("=" * 80)
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No relevant documentation found."
        
        # Método 1: Obtener los chunks más relevantes por similitud vectorial
        logger.info(f"Buscando chunks por similitud vectorial (match_count={MAX_CHUNKS_RETURNED})")
        result = ctx.deps.supabase.rpc(
            'match_visa_mastercard_v5',
            {
                'query_embedding': query_embedding,
                'match_count': MAX_CHUNKS_RETURNED
                #'filter': {"similarity_threshold": 0.5}  # Filtro vacío
            }
        ).execute()
        
        if not result.data:
            logger.info("No relevant documentation found for the query via vector search.")
            # En vez de fallar aquí, seguimos adelante con otros métodos
        
        logger.info(f"Encontrados {len(result.data) if result.data else 0} chunks por similitud vectorial")
        for i, doc in enumerate(result.data or []):
            logger.debug(f"Chunk vectorial #{i+1}: {doc['title']} (similarity: {doc.get('similarity', 'N/A')})")
        
        formatted_chunks = []
        matched_ids = set()  # Para evitar duplicados entre métodos
        cluster_ids = set()  # Para almacenar los cluster_ids de los chunks relevantes
        
        # Método 1 (continuación): Procesar los resultados de búsqueda vectorial
        for doc in (result.data or []):
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            matched_ids.add(doc.get('id'))
            
            # Extraer el cluster_id si existe en los metadatos
            if 'metadata' in doc and doc['metadata'] and 'cluster_id' in doc['metadata']:
                cluster_id = doc['metadata'].get('cluster_id')
                if cluster_id is not None and cluster_id != -1:
                    cluster_ids.add(cluster_id)
        
        logger.info(f"Identificados {len(cluster_ids)} clusters diferentes: {cluster_ids}")
        
        # Método 2: Buscar chunks adicionales por cluster
        additional_chunks = []
        for cluster_id in cluster_ids:
            # Buscar más chunks del mismo cluster
            logger.info(f"Buscando chunks adicionales para el cluster_id={cluster_id}")
            cluster_result = ctx.deps.supabase.rpc(
                'match_visa_mastercard_v5_by_cluster',
                {
                    'cluster_id': cluster_id,
                    'match_count': 9  # Limitamos a 6 chunks adicionales por cluster
                }
            ).execute()
            
            cluster_chunks_count = 0
            if cluster_result.data:
                for doc in cluster_result.data:
                    doc_id = doc.get('id')
                    # Solo añadir si no está ya incluido
                    if doc_id not in matched_ids:
                        matched_ids.add(doc_id)
                        cluster_chunks_count += 1
                        additional_text = f"""
# {doc['title']} (Del mismo tema que otros resultados)

{doc['content']}
"""
                        additional_chunks.append(additional_text)
                        logger.debug(f"Chunk por cluster #{cluster_chunks_count}: {doc['title']} (cluster_id: {cluster_id})")
            
            logger.info(f"Recuperados {cluster_chunks_count} chunks adicionales del cluster {cluster_id}")
        
        # Método 3: Siempre ejecutar BM25 como complemento, no solo como fallback
        bm25_chunks = []
        try:
            # Obtenemos un número razonable de documentos para BM25
            bm25_limit = 15  # Limitamos la cantidad para no exceder el presupuesto total
            logger.info(f"Ejecutando búsqueda léxica BM25 (complementaria)")
            bm25_result = ctx.deps.supabase.table("visa_mastercard_v5").select("id, title, content").execute()
            bm25_docs = bm25_result.data
            
            if bm25_docs:
                # Verificar si NLTK tiene los recursos necesarios y descargarlos correctamente
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    # Usamos download con quiet=True para evitar mensajes innecesarios
                    # y añadimos el parámetro download_dir para asegurar que se descarga en un lugar accesible
                    nltk.download('punkt', quiet=True, download_dir=nltk.data.path[0])
                
                corpus = []
                id_map = []
                full_docs = []
                
                for doc in bm25_docs:
                    text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
                    tokens = word_tokenize(text.lower())
                    corpus.append(tokens)
                    id_map.append(doc.get('id'))
                    full_docs.append(doc)
                
                query_tokens = word_tokenize(user_query.lower())
                bm25 = BM25Okapi(corpus)
                scores = bm25.get_scores(query_tokens)
                
                # Encontrar los mejores documentos por BM25
                best_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_limit]
                bm25_count = 0
                
                for i in best_indices:
                    if scores[i] > 0:  # Solo si hay alguna relevancia
                        doc_id = id_map[i]
                        # Evitar duplicados que ya hayamos recuperado por otros métodos
                        if doc_id not in matched_ids:
                            matched_ids.add(doc_id)
                            doc = full_docs[i]
                            bm25_count += 1
                            
                            bm25_text = f"""
# {doc.get('title', '')} (Coincidencia de términos exactos)

{doc.get('content', '')}
"""
                            bm25_chunks.append(bm25_text)
                
                logger.info(f"Recuperados {bm25_count} chunks adicionales con BM25")
                
        except Exception as e:
            logger.error(f"Error en la recuperación BM25 complementaria: {e}")
        
        # Combinar todos los chunks
        all_chunks = formatted_chunks + additional_chunks + bm25_chunks
        
        # Verificar si tenemos algún resultado
        if not all_chunks:
            logger.info("No relevant documentation found through any method.")
            return "No relevant documentation found."
        
        logger.info(f"RESUMEN: {len(formatted_chunks)} chunks por similitud vectorial + {len(additional_chunks)} chunks por cluster + {len(bm25_chunks)} chunks por BM25 = {len(all_chunks)} chunks en total")
        
        try:
                        
            # Aplicamos reranking solo si hay suficientes chunks para que sea útil
            if len(all_chunks) > 3:
                logger.info("Aplicando reranking con LLM...")
                logger.info(f"Usando modelo: {llm}")
                all_chunks = await rerank_chunks_with_llm(ctx, user_query, all_chunks)
                logger.info("Reranking completado, chunks reordenados por relevancia")

                # Limitar el número de chunks a los N más relevantes
                max_chunks_to_keep = 8  # Puedes ajustar este número según tus necesidades
                if len(all_chunks) > max_chunks_to_keep:
                    logger.info(f"Limitando de {len(all_chunks)} a {max_chunks_to_keep} chunks después del reranking")
                    all_chunks = all_chunks[:max_chunks_to_keep]
        except Exception as e:
            logger.warning(f"No se pudo aplicar reranking con LLM: {e}")
            # Si el reranking falla, continuamos con el orden original de chunks
        
        combined_text = "\n\n---\n\n".join(all_chunks)
        total_tokens = count_tokens(combined_text, llm)
        logger.info(f"Total tokens en todos los chunks: {total_tokens}")
        
        # Truncar si es necesario
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info(f"El contenido combinado excede el límite de tokens ({total_tokens} > {MAX_TOTAL_TOKENS}). Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            chunks_included = 0
            for chunk in all_chunks:
                chunk_tokens = count_tokens(chunk, llm)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens, llm)
                    truncated_chunks.append(truncated_chunk)
                    chunks_included += 1
                    logger.debug(f"Chunk #{chunks_included} truncado para caber en el límite de tokens")
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
                    chunks_included += 1
            
            combined_text = "\n\n---\n\n".join(truncated_chunks)
            logger.info(f"Después de truncar: {chunks_included} chunks incluidos, {count_tokens(combined_text, llm)} tokens totales")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

#@ai_expert.tool
#async def cross_reference_information(ctx: RunContext[AIDeps], primary_topic: str, related_topics: List[str] = None) -> str:
#    """
#    Conecta la consulta regulatoria con contenido relacionado almacenado en la base de datos,
#    asegurando la coherencia y el contexto entre diferentes normativas.
#    
#    Args:
#        primary_topic: El tema principal de la consulta
#        related_topics: Lista opcional de temas relacionados para la referencia cruzada
#    """
#    logger.info("HERRAMIENTA INVOCADA: cross_reference_information")
#    logger.info(f"Tema principal: {primary_topic}")
#    if related_topics:
#        logger.info(f"Temas relacionados: {related_topics}")
#    else:
#        logger.info("No se especificaron temas relacionados")
    
#    try:
#        # Implementación pendiente
#        return "Funcionalidad de referencia cruzada en desarrollo."
#    except Exception as e:
#        logger.error(f"Error en cross_reference_information: {e}")
#        return f"Error en la referencia cruzada de información: {str(e)}"

#@ai_expert.tool
#async def generate_compliance_report(ctx: RunContext[AIDeps], compliance_output: str, template: str = None) -> str:
    # Verifica que se proporcione un output válido
#    if not compliance_output or compliance_output.strip() == "":
#        return "Por favor, proporciona el output completo del agente de Compliance para generar el informe."
    
    # Combina el output de Compliance y la plantilla (si existe)
#    query = compliance_output
#    if template:
#        query += f"\n\nUtiliza la siguiente plantilla para el reporte:\n{template}"
    
    # Llama al agente de Reports sin el parámetro 'tool_name'
#    report_result = await report_agent.run(
#        query,
#        deps=ReportDeps(openai_client=ctx.deps.openai_client),
#        usage=ctx.usage
#    )
#    return report_result.data

@ai_expert.tool
async def perform_gap_analysis(ctx: RunContext[AIDeps], policy_text: str) -> str:
    """
    Realiza un análisis GAP de la política proporcionada en comparación con la normativa de protección de datos.
    
    El análisis debe cumplir con las mejores prácticas de un GAP analysis, incluyendo:
      - **Resumen del estado actual:** Descripción breve de la política evaluada.
      - **Identificación de brechas:** Enumeración de los aspectos en que la política no cumple con la normativa.
      - **Impacto:** Evaluación del impacto de cada brecha identificada.
      - **Recomendaciones:** Propuestas de acciones específicas para cerrar cada brecha.
      - **Priorización:** Orden de prioridad para la implementación de las acciones recomendadas.
    
    **Política a evaluar:**
    {policy_text}
    
    Por favor, genera un análisis GAP detallado y estructurado en formato Markdown.
    """
    logger.info("HERRAMIENTA INVOCADA: perform_gap_analysis")
    logger.info(f"Longitud del texto de política a evaluar: {len(policy_text)} caracteres")
    
    try:
        prompt = f"""Realiza un análisis GAP detallado (mínimo 5.000 palabras) de la siguiente política en comparación con la normativa de protección de datos. El análisis debe cumplir con las mejores prácticas de un GAP analysis, que incluya:
        
        1. **Resumen del estado actual:** Una breve descripción de la política evaluada.
        2. **Identificación de brechas:** Enumeración y descripción de los aspectos donde la política no cumple con la normativa.
        3. **Impacto:** Evaluación del impacto potencial de cada brecha identificada.
        4. **Área impactada:** Identificación del área o departamento afectado por cada brecha.
        5. **Recomendaciones:** Propuestas de acciones específicas para cerrar cada brecha.
        6. **Priorización:** Orden de prioridad de las acciones recomendadas.
        
        Asegúrate de estructurar el análisis de forma clara y ordenada, utilizando encabezados y listas cuando sea apropiado.
        Asegurate de asignar un identificador (ID) a cada GAP detectado.
        **Política a evaluar:**
        {policy_text}
        """
        
        logger.info("Generando análisis GAP usando el modelo LLM")
        
        # Utilizar el modelo para generar el análisis GAP
        response = await ctx.deps.openai_client.chat.completions.create(
                model=llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=3000
            )
            
        # Extraer el contenido de la respuesta
        gap_analysis = response.choices[0].message.content
        
        logger.info(f"GAP analysis completado con éxito. Longitud de la respuesta: {len(gap_analysis)} caracteres")
        return gap_analysis
            
    except Exception as e:
        logger.error(f"Error realizando GAP analysis: {str(e)}")
        return f"Lo siento, ocurrió un error al realizar el análisis de brechas: {str(e)}"


#@ai_expert.tool
#async def translate_response(ctx: RunContext[AIDeps], content: str, target_language: str) -> str:
#    """
#    Traduce el contenido proporcionado al idioma objetivo especificado.
#    
#    Args:
#        content: El texto a traducir
#        target_language: El idioma objetivo (ej. "english", "spanish")
    
#    Returns:
#        El texto traducido al idioma objetivo
#    """
#    logger.info("HERRAMIENTA INVOCADA: translate_response")
#    logger.info(f"Traduciendo contenido al idioma: {target_language}")
    
#    try:
#        # Usamos el modelo GPT para realizar la traducción
#        prompt = f"""Traduce el siguiente texto al idioma {target_language}, 
#        manteniendo el formato, listas, viñetas y estructura. El texto debe quedar 
#        fluido y natural en el idioma objetivo. Conserva cualquier terminología técnica 
#        especializada adaptándola adecuadamente al idioma destino:

#        {content}"""
        
#        response = await ctx.deps.openai_client.chat.completions.create(
#            model=llm,
#            messages=[{"role": "user", "content": prompt}],            
#        )
        
#        translated_content = response.choices[0].message.content
#        logger.info(f"Traducción completada. Longitud del texto traducido: {len(translated_content)} caracteres")
        
#        return translated_content
#    except Exception as e:
#        logger.error(f"Error en la traducción: {str(e)}")
#        return f"Error en la traducción: {str(e)}"


# ===================== FIN DE LAS HERRAMIENTAS DEL AGENTE =====================