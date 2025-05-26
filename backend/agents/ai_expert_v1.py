# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import feedparser
import tiktoken
import time

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Any, Optional, Set
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import deque
from contextlib import asynccontextmanager
# Eliminamos esta importación que causa el problema:
# from agents.report_agent import report_agent, ReportDeps  # Importa el agente de Reports
from utils.reranking_v1 import rerank_chunks
from app.core.config import settings
from utils.utils import count_tokens, truncate_text
from agents.understanding_query import QueryInfo
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_current_query_info = None

MAX_TOTAL_TOKENS = 100000
MAX_CHUNKS_RETURNED = 20

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

Eres "AgentIA", un agente especializado en normativas y regulaciones con acceso EXCLUSIVO a documentación legal específica en tu base de datos.

OBJETIVO PRINCIPAL
Proporcionas análisis jurídicos precisos basados ÚNICAMENTE en la documentación disponible en tu sistema, manteniendo un tono profesional y citando fuentes específicas encontradas.

METODOLOGÍA DE TRABAJO
1. **Recuperación de información**: Utiliza `retrieve_relevant_documentation` para obtener contenido relevante sobre la consulta
2. **Análisis**: Examina EXCLUSIVAMENTE la documentación recuperada para identificar elementos aplicables
3. **Respuesta estructurada**: Presenta la información de forma clara y bien organizada

ESTRUCTURA DE RESPUESTA
- Introducción breve conectando la consulta con el marco normativo encontrado en la documentación
- Citas directas de la documentación usando formato de bloque:
  > [Título del documento] — Artículo X (Jurisdicción, fecha si disponible)
  > «Texto relevante...»
- Análisis jurídico con puntos numerados o viñetas basado en el contenido recuperado
- Conclusiones respaldadas por las fuentes citadas de la documentación

NORMAS ESTRICTAS DE CALIDAD
- **SOLO** citar artículos y documentos específicos encontrados en la documentación recuperada por la herramienta
- **NUNCA** hacer referencia a normativas, leyes, artículos o regulaciones que no aparezcan explícitamente en la documentación recuperada
- **NUNCA** usar conocimiento general sobre leyes o regulaciones que no estén en la documentación proporcionada
- Si algún aspecto no está cubierto en la documentación disponible, indicarlo claramente: "Esta información no se encuentra disponible en la documentación consultada"
- Evitar interpretaciones especulativas no respaldadas por el texto recuperado
- Mantener precisión técnica en terminología jurídica basada en la documentación

MANEJO DE LIMITACIONES
- Si la herramienta de documentación no responde o falla, responder: "No fue posible acceder a la documentación necesaria para responder esta consulta. Por favor, intente nuevamente."
- Si la documentación recuperada es insuficiente para algún aspecto, especificar: "La documentación consultada no contiene información específica sobre [aspecto específico]"
- **PROHIBIDO**: inventar referencias, citar normativas no encontradas en la documentación, o usar conocimiento externo no verificado en la base de datos

VERIFICACIÓN OBLIGATORIA
Antes de mencionar cualquier normativa, ley, artículo o regulación específica, verificar que aparezca explícitamente en la documentación recuperada por la herramienta retrieve_relevant_documentation.

Siempre concluir con: *"Esta respuesta se basa exclusivamente en la documentación consultada y no constituye asesoramiento legal definitivo."*


"""


ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2
)

# -------------------- Herramientas del agente --------------------

async def debug_run_agent(user_query: str, deps: AIDeps, query_info: Optional[QueryInfo] = None):
    """
    Ejecuta el agente de compliance con logging adicional.
    
    Args:
        user_query: La consulta del usuario
        deps: Las dependencias necesarias para el agente
        query_info: Información de análisis de la consulta (opcional)
    """
    logger.debug("Voy a llamar al agente con la query: %s", user_query)
    
    # Creamos una variable global temporal para almacenar query_info
    global _current_query_info
    _current_query_info = query_info
    
    try:
        # Asegurarnos de NO pasar query_info o context como parámetro
        response = await ai_expert.run(
            user_query,
            deps=deps
        )
        
        # Limpiamos la variable global
        _current_query_info = None
        
        # RunResult tiene un método usage() en lugar de get()
        usage_info = response.usage()
        logger.info("Uso de tokens en la consulta: %s", usage_info)
        
        return response
    except Exception as e:
        # Limpiamos la variable global incluso en caso de error
        _current_query_info = None
        # Re-lanzar la excepción para que pueda ser manejada adecuadamente
        raise e

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

# Implementación de la función get_cluster_chunks que falta
async def get_cluster_chunks(ctx, cluster_ids, matched_ids):
    """
    Recupera chunks adicionales por cluster.
    
    Args:
        ctx: El contexto del agente con las dependencias
        cluster_ids: Conjunto de IDs de clusters a buscar
        matched_ids: Conjunto de IDs de documentos ya recuperados para evitar duplicados
    
    Returns:
        Tuple[List[str], Set[int]]: Lista de chunks de texto y conjunto actualizado de IDs coincidentes
    """
    start_time = time.time()
    all_cluster_chunks = []
    new_matched_ids = matched_ids.copy()
    
    # Para cada cluster, lanzamos una búsqueda en paralelo
    async def get_chunks_for_cluster(cluster_id):
        cluster_start_time = time.time()
        logger.info(f"Buscando chunks adicionales para el cluster_id={cluster_id}")
        try:
            cluster_result = ctx.deps.supabase.rpc(
                'match_pd_mex_by_cluster',
                {
                    'cluster_id': cluster_id,
                    'match_count': 5
                }
            ).execute()
            
            cluster_chunks = []
            local_matched_ids = set()
            
            if cluster_result.data:
                for doc in cluster_result.data:
                    doc_id = doc.get('id')
                    # Solo añadir si no está ya incluido en los IDs originales
                    if doc_id not in matched_ids:
                        local_matched_ids.add(doc_id)
                        cluster_chunks.append((doc_id, f"""
{doc['content']}
"""))
            
            cluster_elapsed_time = time.time() - cluster_start_time
            logger.info(f"Tiempo para cluster {cluster_id}: {cluster_elapsed_time:.2f}s - Encontrados: {len(cluster_chunks)} chunks")
            return cluster_chunks, local_matched_ids
        except Exception as e:
            logger.error(f"Error recuperando chunks para cluster {cluster_id}: {e}")
            return [], set()
    
    # Ejecutar búsquedas de clusters en paralelo
    if cluster_ids:
        # Crear las tareas para cada cluster
        cluster_search_tasks = [get_chunks_for_cluster(cid) for cid in cluster_ids]
        
        # Ejecutar todas las tareas en paralelo y esperar los resultados
        cluster_results = await asyncio.gather(*cluster_search_tasks)
        
        # Procesar resultados de forma segura
        for chunks_and_ids in cluster_results:
            chunks, local_ids = chunks_and_ids
            for doc_id, chunk_text in chunks:
                if doc_id not in new_matched_ids:  # Verificación adicional para evitar duplicados
                    new_matched_ids.add(doc_id)
                    all_cluster_chunks.append(chunk_text)
        
        logger.info(f"Recuperados {len(all_cluster_chunks)} chunks adicionales de {len(cluster_ids)} clusters")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Tiempo total de búsqueda por clusters: {elapsed_time:.2f}s")
    return all_cluster_chunks, new_matched_ids

@ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AIDeps], user_query: str, query_info: Optional[QueryInfo] = None) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Aprovecha la información de Query Understanding si está disponible.
    
    Args:
        user_query: Consulta original del usuario
        query_info: Información de análisis de la consulta (opcional)
    """
    start_time_total = time.time()
    logger.info("HERRAMIENTA INVOCADA: retrieve_relevant_documentation")
    logger.info(f"Usando modelo: {llm}")

    # Si no se pasó query_info como parámetro, intentamos obtenerlo de la variable global
    global _current_query_info
    if query_info is None and _current_query_info is not None:
        query_info = _current_query_info
        logger.info("Utilizando información de Query Understanding obtenida de la variable global")

    try:
        # Determinar qué consulta usar para la búsqueda
        search_query = user_query
        
        # Si tenemos información de Query Understanding, usamos la consulta optimizada
        if query_info:
            logger.info("Utilizando información de Query Understanding para mejorar la búsqueda")
            
            # Usar la consulta optimizada para búsqueda si está disponible
            if query_info.search_query:
                search_query = query_info.search_query
                logger.info(f"Usando consulta optimizada para búsqueda: {search_query[:100]}..." if len(search_query) > 100 else search_query)
            # Si no hay consulta optimizada pero sí expandida, usamos esa
            elif query_info.expanded_query:
                search_query = query_info.expanded_query
                logger.info(f"Usando consulta expandida: {search_query[:100]}..." if len(search_query) > 100 else search_query)
            
            # Log de información adicional disponible
            logger.info(f"Información adicional de la consulta:")
            logger.info(f"  Intención principal: {query_info.main_intent}")
            logger.info(f"  Complejidad: {query_info.complexity}")
            logger.info(f"  Entidades detectadas: {[f'{e.type}:{e.value}' for e in query_info.entities]}")
            logger.info(f"  Palabras clave: {[k.word for k in query_info.keywords]}")
        
        logger.info(f"Generando embedding para la consulta de búsqueda...")
        logger.info("=" * 80)
        
        # Primero obtenemos el embedding de la consulta (esto es un prerequisito para las búsquedas)
        start_time_embedding = time.time()
        query_embedding = await get_embedding(search_query, ctx.deps.openai_client)
        embedding_time = time.time() - start_time_embedding
        logger.info(f"Tiempo para generar embedding: {embedding_time:.2f}s")
        
        if not any(query_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No relevant documentation found."
        
        # Definimos las funciones para cada método de búsqueda
        
        async def get_vector_chunks():
            """Recupera chunks por similitud vectorial."""
            start_time = time.time()
            logger.info(f"Buscando chunks por similitud vectorial (match_count={MAX_CHUNKS_RETURNED})")
            try:
                result = ctx.deps.supabase.rpc(
                    'match_pd_mex',
                    {
                        'query_embedding': query_embedding,
                        'match_count': MAX_CHUNKS_RETURNED
                    }
                ).execute()
                
                chunks = []
                matched_ids = set()
                cluster_ids = set()
                
                if result.data:
                    logger.info(f"Encontrados {len(result.data)} chunks por similitud vectorial")
                    
                    for doc in result.data:
                        chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                        chunks.append(chunk_text)
                        matched_ids.add(doc.get('id'))
                        
                        # Extraer cluster_ids
                        if 'metadata' in doc and doc['metadata'] and 'cluster_id' in doc['metadata']:
                            cluster_id = doc['metadata'].get('cluster_id')
                            if cluster_id is not None and cluster_id != -1:
                                cluster_ids.add(cluster_id)
                else:
                    logger.info("No relevant documentation found via vector search.")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Tiempo de búsqueda vectorial: {elapsed_time:.2f}s")
                return chunks, matched_ids, cluster_ids
            except Exception as e:
                logger.error(f"Error en búsqueda vectorial: {e}")
                return [], set(), set()
        
        async def get_bm25_chunks(matched_ids):
            """
            Recupera chunks usando BM25 con EXACTAMENTE los mismos filtros que match_pd_mex RPC.
            """
            start_time = time.time()
            logger.info(f"Ejecutando búsqueda léxica BM25 (complementaria)")
            bm25_chunks = []
            new_matched_ids = matched_ids.copy()
            
            try:
                bm25_limit = 15
                # Incluir metadata en la consulta
                bm25_result = ctx.deps.supabase.table("pd_mex").select("""
                    id, title, summary, content, metadata""").execute()
                
                # APLICAR EXACTAMENTE LA MISMA LÓGICA QUE LA FUNCIÓN RPC
                filtered_docs = []
                for doc in bm25_result.data:
                    regulatory_doc = doc.get('regulatory_documents')
                    doc_metadata = doc.get('metadata', {}) or {}  # Asegurar que no sea None
                    chunk_status = doc_metadata.get('status')
                    
                    # EXACTAMENTE los mismos 3 casos que en match_pd_mex RPC
                    is_vigente = (
                        # Caso 1: Priorizar el status del documento principal si existe
                        (regulatory_doc and regulatory_doc.get('status') == 'vigente') 
                        or 
                        # Caso 2: Si no hay documento principal, verificar status en metadata del chunk
                        (regulatory_doc is None and chunk_status == 'vigente')
                        or
                        # Caso 3: Si no hay información de status en ningún lado, incluir el chunk
                        (regulatory_doc is None and chunk_status is None)
                    )
                    
                    if is_vigente:
                        filtered_docs.append(doc)
                    else:
                        # Log para debug
                        doc_status = regulatory_doc.get('status') if regulatory_doc else 'None'
                        logger.debug(f"BM25 excluyó chunk {doc.get('id')}: doc_status={doc_status}, chunk_status={chunk_status}")
                
                logger.info(f"BM25: {len(bm25_result.data)} chunks iniciales → {len(filtered_docs)} chunks después del filtro de vigencia")
                
                if filtered_docs:
                    # Verificar NLTK
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True, download_dir=nltk.data.path[0])
                    
                    corpus = []
                    id_map = []
                    full_docs = []
                    
                    for doc in filtered_docs:  # ← Usar filtered_docs en lugar de bm25_docs
                        # Para BM25, usar el contenido completo incluyendo metadata
                        text_parts = [
                            doc.get('title', ''),
                            doc.get('summary', ''),
                            doc.get('content', ''),
                            str(doc.get('metadata', ''))  # Convertir metadata a string para búsqueda
                        ]
                        text = ' '.join(filter(None, text_parts))  # Filtrar partes vacías
                        
                        tokens = word_tokenize(text.lower())
                        corpus.append(tokens)
                        id_map.append(doc.get('id'))
                        full_docs.append(doc)
                    
                    # Usar palabras clave de Query Understanding si están disponibles
                    if query_info and query_info.keywords:
                        important_keywords = [k.word for k in query_info.keywords if k.importance > 0.7]
                        if important_keywords:
                            logger.info(f"Usando palabras clave de alta importancia para BM25: {important_keywords}")
                            query_tokens = word_tokenize(" ".join(important_keywords).lower())
                        else:
                            query_tokens = word_tokenize(search_query.lower())
                    else:
                        query_tokens = word_tokenize(search_query.lower())
                    
                    bm25 = BM25Okapi(corpus)
                    scores = bm25.get_scores(query_tokens)
                    
                    best_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_limit]
                    
                    for i in best_indices:
                        if scores[i] > 0:
                            doc_id = id_map[i]
                            if doc_id not in new_matched_ids:
                                new_matched_ids.add(doc_id)
                                doc = full_docs[i]
                                
                                summary = doc.get('summary', '')
                                summary_section = f"\nResumen: {summary}\n" if summary else ""

                                bm25_text = f"""
        # {doc.get('title', '')} (Coincidencia de términos exactos)
        {summary_section}
        {doc.get('content', '')}
        """
                                bm25_chunks.append(bm25_text)
                                logger.debug(f"BM25 incluyó chunk {doc_id} con score {scores[i]:.3f}")
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Recuperados {len(bm25_chunks)} chunks adicionales con BM25 en {elapsed_time:.2f}s")
            except Exception as e:
                logger.error(f"Error en la recuperación BM25: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                elapsed_time = time.time() - start_time
                logger.error(f"Tiempo hasta error BM25: {elapsed_time:.2f}s")
            
            return bm25_chunks
        
        # Si tenemos entidades de Query Understanding, podemos hacer una búsqueda adicional por entidades
        async def get_entity_based_chunks(matched_ids):
            """
            Recupera chunks basados en las entidades detectadas por Query Understanding.
            Solo se ejecuta si hay información de Query Understanding disponible.
            """
            if not query_info or not query_info.entities:
                return []
            
            start_time = time.time()
            logger.info(f"Ejecutando búsqueda basada en entidades")
            entity_chunks = []
            new_matched_ids = matched_ids.copy()
            
            try:
                # Extraer entidades relevantes (priorizar regulation, program, process)
                priority_types = ['regulation', 'program', 'process', 'technical_requirement']
                relevant_entities = [e for e in query_info.entities if e.type in priority_types]
                
                if not relevant_entities:
                    logger.info("No hay entidades de alta prioridad para búsqueda")
                    return []
                
                # Construir condiciones para la consulta SQL
                entity_conditions = []
                for entity in relevant_entities:
                    # Escapar valor para SQL y convertir a minúsculas para comparación insensible a mayúsculas
                    value = entity.value.lower().replace("'", "''")
                    entity_conditions.append(f"(LOWER(content) LIKE '%{value}%' OR LOWER(title) LIKE '%{value}%')")
                
                if not entity_conditions:
                    return []
                
                # Combinar condiciones con OR
                where_clause = " OR ".join(entity_conditions)
                
                # Ejecutar consulta en Supabase
                entity_query = ctx.deps.supabase.table("pd_mex").select("id, title, summary, content, article_references").filter(where_clause, False).execute()
                
                if entity_query.data:
                    for doc in entity_query.data:
                        doc_id = doc.get('id')
                        if doc_id not in new_matched_ids:
                            new_matched_ids.add(doc_id)
                            
                            summary = doc.get('summary', '')
                            summary_section = f"\nResumen: {summary}\n" if summary else ""
                            
                            entity_text = f"""
# {doc.get('title', '')} (Coincidencia por entidad específica)
{summary_section}
{doc.get('content', '')}
"""
                            entity_chunks.append(entity_text)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Recuperados {len(entity_chunks)} chunks adicionales por entidades en {elapsed_time:.2f}s")
                return entity_chunks
                
            except Exception as e:
                logger.error(f"Error en la búsqueda por entidades: {e}")
                return []
        
        # Ejecutar búsqueda vectorial primero (necesitamos los cluster_ids)
        start_search_time = time.time()
        vector_chunks, matched_ids, cluster_ids = await get_vector_chunks()
        vector_time = time.time() - start_search_time
        
        # Luego ejecutamos en paralelo las búsquedas complementarias
        start_parallel_time = time.time()
        
        # Tareas asíncronas para las búsquedas complementarias
        tasks = [
            get_cluster_chunks(ctx, cluster_ids, matched_ids),  # Pasamos el contexto ctx como primer argumento
            get_bm25_chunks(matched_ids)
        ]
        
        # Si tenemos información de Query Understanding, agregamos la búsqueda por entidades
        if query_info and query_info.entities:
            tasks.append(get_entity_based_chunks(matched_ids))
        
        # Ejecutar todas las búsquedas en paralelo
        parallel_results = await asyncio.gather(*tasks)
        
        # Extraer resultados
        cluster_chunks_result = parallel_results[0]
        bm25_chunks = parallel_results[1]
        cluster_chunks, updated_matched_ids = cluster_chunks_result
        
        # Extraer chunks basados en entidades si existen
        entity_chunks = []
        if query_info and query_info.entities and len(parallel_results) > 2:
            entity_chunks = parallel_results[2]
        
        parallel_time = time.time() - start_parallel_time
        logger.info(f"Tiempo de búsquedas paralelas: {parallel_time:.2f}s")
        
        # Combinar todos los chunks
        all_chunks = vector_chunks + cluster_chunks + bm25_chunks + entity_chunks
        
        # Verificar si tenemos algún resultado
        if not all_chunks:
            logger.info("No relevant documentation found through any method.")
            return "No relevant documentation found."
        
        logger.info(f"RESUMEN: {len(vector_chunks)} chunks por similitud vectorial + {len(cluster_chunks)} chunks por cluster + {len(bm25_chunks)} chunks por BM25 + {len(entity_chunks)} chunks por entidades = {len(all_chunks)} chunks en total")
        
        # Tiempo para reranking
        start_rerank_time = time.time()
        try:
            # Aplicamos reranking con información de Query Understanding si está disponible
            if len(all_chunks) > 3:
                logger.info("Aplicando reranking con LLM...")
                logger.info(f"Usando modelo: {llm}")
                
                # Construir prompt de reranking enriquecido con información de Query Understanding
                reranking_query = search_query
                if query_info:
                    # Construir contexto enriquecido para el reranking
                    intent_context = f"Intención principal: {query_info.main_intent}" if query_info.main_intent else ""
                    entity_context = f"Entidades importantes: {', '.join([e.value for e in query_info.entities[:5]])}" if query_info.entities else ""
                    keyword_context = f"Palabras clave: {', '.join([k.word for k in query_info.keywords[:5]])}" if query_info.keywords else ""
                    
                    reranking_query = f"{search_query}\n\nContexto adicional:\n{intent_context}\n{entity_context}\n{keyword_context}"
                
                # Usar el reranking mejorado con el contexto enriquecido
                reranked_chunks = await rerank_chunks(ctx, reranking_query, all_chunks, max_to_rerank=15)
                if reranked_chunks:  # Verificar que el resultado no sea vacío
                    all_chunks = reranked_chunks
                    logger.info("Reranking completado, chunks reordenados por relevancia")
                else:
                    logger.warning("El reranking no produjo resultados, manteniendo orden original")

                # Limitar el número de chunks a los N más relevantes
                max_chunks_to_keep = 8
                if query_info and query_info.complexity == "complex":
                    max_chunks_to_keep = 12  # Para consultas complejas, permitimos más contexto
                
                if len(all_chunks) > max_chunks_to_keep:
                    logger.info(f"Limitando de {len(all_chunks)} a {max_chunks_to_keep} chunks después del reranking")
                    all_chunks = all_chunks[:max_chunks_to_keep]
        except Exception as e:
            logger.warning(f"No se pudo aplicar reranking con LLM: {e}")
            # Agregar más detalles sobre el error para facilitar el debugging
            logger.warning(f"Tipo de error: {type(e).__name__}")
            logger.warning(f"Detalles completos del error: {str(e)}")
            # Si el reranking falla, continuamos con el orden original de chunks
        
        rerank_time = time.time() - start_rerank_time
        logger.info(f"Tiempo de reranking: {rerank_time:.2f}s")
        
        # Proceso final: combinación y truncamiento
        start_final_time = time.time()
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
        
        final_time = time.time() - start_final_time
        total_time = time.time() - start_time_total
        
        # Resumen de tiempos
        logger.info("==== RESUMEN DE TIEMPOS ====")
        logger.info(f"Generación de embedding: {embedding_time:.2f}s")
        logger.info(f"Búsqueda vectorial: {vector_time:.2f}s")
        logger.info(f"Búsquedas paralelas: {parallel_time:.2f}s")
        logger.info(f"Reranking: {rerank_time:.2f}s")
        logger.info(f"Combinación y truncamiento: {final_time:.2f}s")
        logger.info(f"TIEMPO TOTAL: {total_time:.2f}s")
        logger.info("==========================")
        
        # Si tenemos información de Query Understanding, agregamos metadatos al resultado
        if query_info:
            # Agregar un separador con información sobre la consulta
            metadata_header = f"""
=== INFORMACIÓN DE ANÁLISIS DE CONSULTA ===
Intención principal: {query_info.main_intent}
Complejidad: {query_info.complexity}
Entidades detectadas: {[f'{e.type}:{e.value}' for e in query_info.entities]}
Consulta utilizada para búsqueda: {search_query[:100]}..." if len(search_query) > 100 else search_query
===========================================

"""
            combined_text = metadata_header + combined_text
        
        return combined_text
    
    except Exception as e:
        total_time = time.time() - start_time_total
        logger.error(f"Error retrieving documentation en {total_time:.2f}s: {e}")
        return f"Error retrieving documentation: {str(e)}"


# ================== HERRAMIENTA DE GAP ANALYSIS SIMPLIFICADA ==================
# Añadir esta herramienta a tu agente existente (ai_expert_v1.py)

@ai_expert.tool
async def perform_gap_analysis(ctx: RunContext[AIDeps], policy_text: str, focus_areas: Optional[str] = None) -> str:
    """
    Realiza un análisis GAP detallado entre una política interna y la normativa aplicable.
    
    Args:
        policy_text: El texto completo de la política a evaluar
        focus_areas: Áreas específicas a evaluar (opcional): "consentimiento, derechos ARCO, seguridad", etc.
    
    Returns:
        Análisis GAP completo en formato Markdown con brechas identificadas y recomendaciones
    """
    logger.info("HERRAMIENTA INVOCADA: perform_gap_analysis")
    
    try:
        # Construir consulta de búsqueda
        search_query = "normativa protección datos privacidad requisitos obligatorios derechos seguridad"
        if focus_areas:
            search_query += f" {focus_areas}"
        
        logger.info(f"Recuperando documentación normativa para GAP analysis...")
        
        # Reutilizar COMPLETAMENTE la función existente
        regulatory_docs = await retrieve_relevant_documentation(ctx, search_query)
        
        if not regulatory_docs or regulatory_docs == "No relevant documentation found.":
            return "No se pudo recuperar documentación normativa suficiente para realizar el análisis GAP."
        
        # Prompt simplificado sin iconos
        gap_prompt = f"""
Realiza un análisis GAP profesional y detallado (mínimo 4.000 palabras) comparando la política interna con la normativa aplicable.

POLÍTICA A EVALUAR:
{policy_text[:8000]}

NORMATIVA DE REFERENCIA:
{regulatory_docs}

ESTRUCTURA REQUERIDA:

## Resumen Ejecutivo
- Política evaluada: [Identificar tipo y alcance]
- Total de brechas: [Número] (Alto: X, Medio: Y, Bajo: Z)
- Nivel de cumplimiento: [Porcentaje estimado]
- Recomendación principal: [Acción más crítica]

## Análisis Detallado de Brechas

### GAP-001: [Nombre descriptivo]
- **Descripción:** [Qué no cumple la política]
- **Requisito normativo:** [Artículo específico de la documentación]
- **Estado actual:** [Cómo aborda este tema la política actual]
- **Nivel de riesgo:** [Alto/Medio/Bajo] - [Justificación]
- **Área impactada:** [Departamento/función]
- **Recomendación:** [Acción específica con texto sugerido]
- **Esfuerzo:** [Alto/Medio/Bajo] - [Justificación]

[Continuar con GAP-002, GAP-003, etc. - MÍNIMO 8 brechas]

## Matriz de Priorización

| ID | Brecha | Riesgo | Esfuerzo | Prioridad | Plazo |
|----|--------|--------|----------|-----------|--------|
| GAP-001 | [Resumen] | Alto | Medio | 1 | 15 días |
| GAP-002 | [Resumen] | Medio | Bajo | 2 | 30 días |

## Plan de Implementación

### Fase 1: Crítico (0-30 días)
- [Brechas de alto riesgo con acciones específicas]

### Fase 2: Importante (30-90 días)  
- [Brechas de riesgo medio]

### Fase 3: Mejoras (90+ días)
- [Brechas de bajo riesgo y optimizaciones]

## Métricas de Seguimiento
- [KPIs específicos para medir progreso]

INSTRUCCIONES CRÍTICAS:
1. Identificar AL MENOS 8-10 brechas específicas
2. Citar SOLO artículos presentes en la documentación normativa proporcionada
3. Ser específico en recomendaciones (incluir texto exacto a añadir/modificar)
4. Justificar cada nivel de riesgo con criterios objetivos
5. Proporcionar estimaciones realistas de tiempo y esfuerzo
6. Incluir consideraciones prácticas de implementación

Enfócate en crear un análisis accionable y profesional.
"""

        # Ejecutar análisis
        response = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": gap_prompt}],
            temperature=0.1,
            max_tokens=12000
        )
        
        gap_result = response.choices[0].message.content
        
        # Añadir disclaimer simple
        gap_result += "\n\n---\n\n*Este análisis GAP se basa exclusivamente en la documentación normativa consultada y no constituye asesoramiento legal definitivo.*"
        
        logger.info("GAP analysis completado exitosamente")
        return gap_result
        
    except Exception as e:
        logger.error(f"Error en análisis GAP: {e}")
        return f"Error realizando análisis GAP: {str(e)}"