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
from utils.reranking_v1 import rerank_chunks
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

Eres un experto en normativas de VISA y Mastercard, operando como un agente de inteligencia artificial con acceso a documentación completa y actualizada.

Tu misión es responder a todas las consultas proporcionando información precisa, detallada y bien estructurada, enfocada exclusivamente en el ecosistema de pagos y el cumplimiento normativo. Al responder, debes incluir todos los detalles y el contexto relevantes, utilizando listas numeradas o viñetas para desglosar procesos complejos.

INSTRUCCIONES CLAVE:
• Antes de responder cualquier consulta, DEBES utilizar la herramienta retrieve_relevant_documentation para extraer y resumir los fragmentos de documentación más relevantes de VISA y Mastercard.
• Si se solicita información sobre las fuentes utilizadas en tu respuesta, utiliza la herramienta list_used_documentation para mostrar específicamente los documentos que fueron utilizados para generar la respuesta actual.
• La información recuperada debe mostrarse siempre en el formato exacto del "chunk" extraído de la base de datos vectorial.
• Sé sincero e indica si la documentación disponible no abarca todos los aspectos necesarios.
• No solicites información adicional ni te disculpes por la falta de detalles; infiere y proporciona la respuesta más completa y precisa.
• Cuando sea necesario, utiliza también la herramienta "perform_gap_analysis" para llevar a cabo un análisis de brechas en la política proporcionada.
• NUNCA debes responder a una consulta sin haber ejecutado primero la herramienta retrieve_relevant_documentation, excepto cuando se solicite específicamente una lista de documentación, en cuyo caso utilizarás list_documentation_pages.


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
            'match_visa_mastercard_v7',
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
                'match_visa_mastercard_v7_by_cluster',
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
            bm25_result = ctx.deps.supabase.table("visa_mastercard_v7").select("id, title, summary, content").execute()
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
                reranked_chunks = await rerank_chunks(ctx, user_query, all_chunks, max_to_rerank=15)
                if reranked_chunks:  # Verificar que el resultado no sea vacío
                    all_chunks = reranked_chunks
                    logger.info("Reranking completado, chunks reordenados por relevancia")
                else:
                    logger.warning("El reranking no produjo resultados, manteniendo orden original")

                # Limitar el número de chunks a los N más relevantes
                max_chunks_to_keep = 8
                if len(all_chunks) > max_chunks_to_keep:
                    logger.info(f"Limitando de {len(all_chunks)} a {max_chunks_to_keep} chunks después del reranking")
                    all_chunks = all_chunks[:max_chunks_to_keep]
        except Exception as e:
            logger.warning(f"No se pudo aplicar reranking con LLM: {e}")
            # Agregar más detalles sobre el error para facilitar el debugging
            logger.warning(f"Tipo de error: {type(e).__name__}")
            logger.warning(f"Detalles completos del error: {str(e)}")
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

@ai_expert.tool
async def list_used_documentation(ctx: RunContext[AIDeps], chunks_content: str = None) -> str:
    """
    Identifica y lista los documentos utilizados en la generación de una respuesta,
    analizando directamente los chunks de documentación proporcionados, teniendo en cuenta
    que estos chunks han pasado por un proceso de reranking.
    
    Args:
        chunks_content: El contenido de los chunks finales devueltos por retrieve_relevant_documentation,
                        después del proceso de reranking. Si no se proporciona, se solicitará al usuario
                        que ejecute primero retrieve_relevant_documentation y proporcione su salida.
    
    Returns:
        str: Lista formateada de documentos utilizados con sus títulos y URLs, ordenados según su
             relevancia determinada por el proceso de reranking.
    """
    logger.info("HERRAMIENTA INVOCADA: list_used_documentation")
    
    if not chunks_content:
        return """
Para listar los documentos utilizados, es necesario proporcionar el contenido de los chunks que fueron utilizados.

Por favor, ejecuta esta herramienta después de retrieve_relevant_documentation y proporciona el resultado 
como parámetro chunks_content. Este resultado ya incluye los chunks reordenados por el proceso de reranking.

Ejemplo de uso:
1. Ejecuta retrieve_relevant_documentation(query)
2. Usa el resultado para ejecutar list_used_documentation(chunks_content=resultado_anterior)
"""
    
    try:
        logger.info(f"Analizando contenido de chunks post-reranking (longitud: {len(chunks_content)} caracteres)")
        
        # Dividir el contenido en chunks individuales (separados por "---")
        individual_chunks = re.split(r'\n\n---\n\n', chunks_content)
        logger.info(f"Identificados {len(individual_chunks)} chunks individuales post-reranking")
        
        # Extraer títulos y mantener el orden de relevancia determinado por el reranking
        document_info = []
        for i, chunk in enumerate(individual_chunks):
            # Buscar el título que comienza con #
            title_match = re.search(r'#\s+([^\n]+?)(?:\s*\([^)]*\))?(?:\n|\r|$)', chunk)
            if title_match:
                title = title_match.group(1).strip()
                # Almacenar título y posición (relevancia) según el reranking
                document_info.append({
                    "title": title,
                    "rank": i + 1  # Posición 1-based para mejor presentación
                })
                logger.debug(f"Chunk #{i+1}: {title}")
        
        if not document_info:
            logger.info("No se encontraron títulos de documentos en los chunks")
            return "No se pudieron identificar documentos en el contenido proporcionado."
        
        # Eliminar duplicados preservando el orden y la posición de primera aparición
        unique_docs = {}
        for doc in document_info:
            if doc["title"] not in unique_docs:
                unique_docs[doc["title"]] = doc["rank"]
        
        logger.info(f"Encontrados {len(unique_docs)} títulos únicos de documentos")
        
        # Consultar la base de datos para obtener información completa
        used_docs_info = []
        
        for title, rank in unique_docs.items():
            # Consultar información adicional (como URLs) de la base de datos
            safe_title = title.replace("'", "''")
            
            # Primero intentamos una búsqueda exacta
            result = ctx.deps.supabase.table("visa_mastercard_v7").select("id, title, url").eq("title", title).limit(1).execute()
            
            # Si no hay resultados exactos, intentamos una búsqueda por similitud
            if not result.data:
                result = ctx.deps.supabase.table("visa_mastercard_v7").select("id, title, url").ilike("title", f"%{safe_title}%").limit(1).execute()
            
            if result.data:
                doc = result.data[0]
                used_docs_info.append({
                    "title": title,  # Usamos el título original extraído
                    "url": doc.get("url", ""),
                    "rank": rank
                })
                logger.debug(f"Encontrada URL para: {title}")
            else:
                # Si no encontramos el documento en la base de datos, incluimos solo el título
                used_docs_info.append({
                    "title": title, 
                    "url": "",
                    "rank": rank
                })
                logger.debug(f"No se encontró URL para: {title}")
        
        # Ordenar por relevancia (rank) - el orden que tenían después del reranking
        used_docs_info.sort(key=lambda x: x["rank"])
        
        # Formatear la respuesta
        formatted_response = "## Documentos utilizados en esta respuesta (ordenados por relevancia):\n\n"
        
        for i, doc in enumerate(used_docs_info, 1):
            doc_entry = f"{i}. **{doc['title']}**"
            if doc["url"]:
                doc_entry += f"\n   URL: {doc['url']}"
            formatted_response += doc_entry + "\n\n"
        
        logger.info(f"Listados {len(used_docs_info)} documentos utilizados, ordenados según relevancia del reranking")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error al analizar documentos utilizados: {str(e)}")
        return f"Error al recuperar la lista de documentos utilizados: {str(e)}"


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