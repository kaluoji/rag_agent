import re
import logging
from typing import List, Any, Tuple
from pydantic_ai import RunContext
import nltk
from nltk.tokenize import word_tokenize
import asyncio
import numpy as np
from rank_bm25 import BM25Okapi
import hashlib
from functools import lru_cache

logger = logging.getLogger(__name__)

# Caché para almacenar resultados de reranking
reranking_cache = {}

# Función para generar una clave de caché basada en consulta y chunks
def generate_cache_key(query: str, chunks: List[str]) -> str:
    """Genera una clave de caché única basada en la consulta y los chunks."""
    # Concatenar consulta y primeros 100 caracteres de cada chunk para formar una huella digital
    content = query + "".join([chunk[:100] for chunk in chunks])
    return hashlib.md5(content.encode()).hexdigest()

# Asegúrate de que NLTK descargue el tokenizador 'punkt'
nltk.download('punkt', quiet=True)

async def prepare_chunk_data(chunks: List[str], openai_client) -> Tuple[List[List[str]], List[Any]]:
    """
    Prepara los datos necesarios para el reranking híbrido:
      - chunk_tokens: lista de tokens (resultado de word_tokenize) para cada chunk.
      - chunk_embeddings: lista de embeddings para cada chunk.
    
    Optimizado para procesar embeddings en lotes y reducir llamadas a la API.
    """
    # Genera tokens para cada chunk
    chunk_tokens = [word_tokenize(chunk.lower()) for chunk in chunks]
    
    # Procesar embeddings en lotes para reducir llamadas a la API
    BATCH_SIZE = 8  # Ajusta este valor según las limitaciones de la API
    chunk_embeddings = []
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        logger.info(f"Procesando lote de embeddings {i//BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        try:
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            chunk_embeddings.extend(batch_embeddings)
        except Exception as e:
            # Si falla el lote, intentar uno por uno como fallback
            logger.warning(f"Error al procesar lote de embeddings: {e}. Intentando uno por uno.")
            for chunk in batch:
                try:
                    single_response = await openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    chunk_embeddings.append(single_response.data[0].embedding)
                except Exception as inner_e:
                    logger.error(f"Error al procesar embedding individual: {inner_e}")
                    # Insertar un vector de ceros como fallback
                    chunk_embeddings.append([0.0] * 1536)  # Dimensión estándar de embedding-3-small
    
    return chunk_tokens, chunk_embeddings

async def evaluate_chunk_relevance(ctx: RunContext[Any], query: str, chunk: str) -> float:
    """
    Evalúa la relevancia de un chunk para la consulta utilizando un LLM.
    Devuelve un puntaje del 0 al 10 basado en pertinencia, precisión y nivel de detalle.
    
    """
    # Extraer título para el logging
    title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
    title = title_match.group(1) if title_match else "Chunk sin título"
    
    truncated_chunk = chunk[:500]
    eval_prompt = f"""
Necesito que evalúes la relevancia del siguiente fragmento de texto en relación con la siguiente consulta, considerando tres criterios clave:

1. Pertinencia global: ¿Cuán directamente responde el fragmento a la consulta?
2. Precisión y exactitud: ¿Contiene el fragmento información exacta y correcta (por ejemplo, datos numéricos, fechas, criterios específicos)?
3. Nivel de detalle (granularidad): ¿Proporciona el fragmento suficientes matices y detalles para responder de forma completa a la consulta?

Consulta: "{query}"

Fragmento a evaluar:
---
{truncated_chunk}
---

Para cada criterio, asigna un puntaje del 0 al 10 (donde 10 es excelente y 0 es inexistente). Luego, calcula un puntaje global ponderado con la siguiente fórmula:
Puntaje Global = 0.4 * (Pertinencia) + 0.3 * (Precisión) + 0.3 * (Detalle)

IMPORTANTE: Responde ÚNICAMENTE con el número del Puntaje Global final (redondeado al entero más cercano), SIN texto adicional. Por ejemplo, si el resultado es 7.5, responde solo con "8".
"""
    try:
        response = await ctx.deps.openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # Modelo económico para evaluación
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=5
        )
        score_text = response.choices[0].message.content.strip()
        # Extrae cualquier número del texto
        match = re.search(r'\d+(\.\d+)?', score_text)
        if match:
            # No agregamos logging aquí, lo haremos en las funciones que llaman a esta
            return float(match.group(0))
        else:
            # Si no se encuentra un número, intenta convertir directamente el texto completo
            try:
                return float(score_text)
            except ValueError:
                logger.warning(f"No se extrajo puntaje numérico de: '{score_text}' para chunk '{title[:30]}...'. Se asigna 0.")
                return 0.0
    except Exception as e:
        logger.error(f"Error en evaluate_chunk_relevance: {e}")
        return 0.0

def normalize(scores: List[float]) -> np.ndarray:
    scores = np.array(scores, dtype=float)
    if scores.max() - scores.min() == 0:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

async def hybrid_rerank_chunks(
    ctx: RunContext[Any],
    query: str,
    chunks: List[str],
    chunk_tokens: List[List[str]],
    chunk_embeddings: List[Any],
    bm25_weight: float = 0.3,
    cosine_weight: float = 0.3,
    llm_weight: float = 0.4,
    max_to_rerank: int = 15
) -> List[str]:
    """
    Realiza un reranking híbrido combinando señales de:
      - BM25 (coincidencia léxica),
      - Similitud coseno (embeddings semánticos) y
      - Evaluación LLM (pertinencia, precisión y detalle).
    """
    try:
        if not chunks:
            logger.warning("No hay chunks para reordenar")
            return []
            
        if len(chunks) <= 1:
            return chunks
            
        # 1. Señal BM25
        bm25_model = BM25Okapi(chunk_tokens)
        query_tokens = word_tokenize(query.lower())
        bm25_scores = bm25_model.get_scores(query_tokens)
        
        # 2. Señal de similitud coseno
        query_embedding_response = await ctx.deps.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding)
        cosine_scores = []
        for emb in chunk_embeddings:
            emb_arr = np.array(emb)
            norm_query = np.linalg.norm(query_embedding)
            norm_emb = np.linalg.norm(emb_arr)
            cosine = np.dot(query_embedding, emb_arr) / (norm_query * norm_emb) if norm_query * norm_emb != 0 else 0
            cosine_scores.append(cosine)
        
        # 3. Señal LLM: se evalúa solo para los primeros max_to_rerank chunks para controlar el coste
        llm_scores = []
        for i, chunk in enumerate(chunks[:max_to_rerank]):
            score = await evaluate_chunk_relevance(ctx, query, chunk)
            
            # Extraer título para el logging
            title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
            title = title_match.group(1) if title_match else f"Chunk {i+1}"
            logger.info(f"Chunk '{title[:50]}...' recibió puntuación LLM: {score}")
            
            llm_scores.append(score)
            
        if len(chunks) > max_to_rerank:
            llm_scores.extend([0] * (len(chunks) - max_to_rerank))
        
        # Normaliza las señales
        bm25_norm = normalize(bm25_scores)
        cosine_norm = normalize(cosine_scores)
        llm_norm = normalize(llm_scores)
        
        # Combina las señales con los pesos definidos
        combined_scores = bm25_weight * bm25_norm + cosine_weight * cosine_norm + llm_weight * llm_norm
        
        # Ordena los índices según el puntaje combinado y retorna los chunks ordenados
        ranked_indices = sorted(range(len(chunks)), key=lambda i: combined_scores[i], reverse=True)
        ranked_chunks = [chunks[i] for i in ranked_indices]
        
        # Muestra las puntuaciones de los 3 mejores chunks
        top5_indices = sorted(range(len(chunks)), key=lambda i: combined_scores[i], reverse=True)[:3]
        logger.info(f"Reranking híbrido completado. Top 5 puntuaciones: {[combined_scores[i] for i in top5_indices]}")
        
        return ranked_chunks
        
    except Exception as e:
        logger.error(f"Error en hybrid_rerank_chunks: {e}")
        # En caso de error, devolvemos los chunks originales sin reordenar
        return chunks

# Función de reranking original (solo LLM) como fallback
async def rerank_chunks_with_llm(ctx: RunContext[Any], query: str, chunks: List[str], max_to_rerank: int = 15) -> List[str]:
    logger.info(f"Iniciando reranking con LLM para {len(chunks)} chunks (máximo a reordenar: {max_to_rerank})")
    if len(chunks) <= 1:
        return chunks
    chunks_to_rerank = chunks[:max_to_rerank]
    remaining_chunks = chunks[max_to_rerank:] if len(chunks) > max_to_rerank else []
    ranked_chunks = []
    for i, chunk in enumerate(chunks_to_rerank):
        title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
        title = title_match.group(1) if title_match else f"Chunk {i+1}"
        eval_prompt = f"""
Necesito que evalúes la relevancia del siguiente fragmento de texto en relación con la siguiente consulta, considerando tres criterios clave:

1. Pertinencia global: ¿Cuán directamente responde el fragmento a la consulta?
2. Precisión y exactitud: ¿Contiene el fragmento información exacta y correcta (por ejemplo, datos numéricos, fechas, criterios específicos)?
3. Nivel de detalle (granularidad): ¿Proporciona el fragmento suficientes matices y detalles para responder de forma completa a la consulta?

Consulta: "{query}"

Fragmento a evaluar:
---
{chunk[:500]}
---

Para cada criterio, asigna un puntaje del 0 al 10. Luego, calcula un puntaje global ponderado:
Puntaje Global = 0.4 * (Pertinencia) + 0.3 * (Precisión) + 0.3 * (Detalle)

Responde únicamente con el número del Puntaje Global final (redondeado al entero más cercano).
"""
        try:
            response = await ctx.deps.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=5
            )
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(re.search(r'\d+(\.\d+)?', score_text).group(0))
                logger.info(f"Chunk '{title}' recibió puntuación: {score}")
                ranked_chunks.append((chunk, score))
            except Exception as e:
                logger.warning(f"No se pudo extraer puntaje numérico de '{score_text}', asignando 0")
                ranked_chunks.append((chunk, 0))
        except Exception as e:
            logger.error(f"Error evaluando chunk {i+1}: {e}")
            ranked_chunks.append((chunk, 0))
    ranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Logging de top puntuaciones
    logger.info(f"Reranking completado. Top 5 puntuaciones: {[score for _, score in ranked_chunks[:5]]}")
    
    reranked_chunks = [chunk for chunk, _ in ranked_chunks]
    return reranked_chunks + remaining_chunks

# Función principal para reranking que decide qué método utilizar
async def rerank_chunks(ctx: RunContext[Any], query: str, chunks: List[str], max_to_rerank: int = 15) -> List[str]:
    """
    Función principal para reordenar chunks según su relevancia para la consulta.
    Esta función es la que debe ser llamada desde cualquier parte del código que necesite reordenar chunks.
    
    Implementa un mecanismo de caché para evitar procesar la misma consulta y chunks múltiples veces.
    """
    logger.info(f"Iniciando reranking para {len(chunks)} chunks")
    
    if not chunks:
        logger.warning("No hay chunks para reordenar")
        return []
        
    if len(chunks) <= 1:
        return chunks
    
    # Generar clave de caché
    cache_key = generate_cache_key(query, chunks)
    
    # Verificar si ya tenemos estos resultados en caché
    if cache_key in reranking_cache:
        logger.info("Resultados encontrados en caché, omitiendo procesamiento de reranking")
        return reranking_cache[cache_key]
    
    try:
        # Preparamos los datos necesarios para el reranking híbrido
        logger.info("Preparando datos para reranking híbrido")
        chunk_tokens, chunk_embeddings = await prepare_chunk_data(chunks, ctx.deps.openai_client)
        
        # Llamamos al reranking híbrido
        logger.info("Ejecutando reranking híbrido")
        ranked_chunks = await hybrid_rerank_chunks(
            ctx, 
            query, 
            chunks, 
            chunk_tokens, 
            chunk_embeddings, 
            max_to_rerank=max_to_rerank
        )
        
        # Guardar resultados en caché
        reranking_cache[cache_key] = ranked_chunks
        
        # Limitar el tamaño de la caché a las 50 consultas más recientes
        if len(reranking_cache) > 50:
            oldest_key = next(iter(reranking_cache))
            del reranking_cache[oldest_key]
            
        return ranked_chunks
    except Exception as e:
        logger.warning(f"Reranking híbrido falló: {e}")
        logger.info("Fallback: Usando reranking con LLM")
        fallback_results = await rerank_chunks_with_llm(ctx, query, chunks, max_to_rerank)
        
        # Guardar resultados de fallback en caché también
        reranking_cache[cache_key] = fallback_results
        return fallback_results