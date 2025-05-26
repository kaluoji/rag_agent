import re
import logging
import time
from typing import List, Any, Tuple, Dict, Optional, Union, Callable
from pydantic_ai import RunContext
import nltk
from nltk.tokenize import word_tokenize
import asyncio
import numpy as np
from rank_bm25 import BM25Okapi
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import json

# Configuración mejorada de logging
logger = logging.getLogger(__name__)

# Caché mejorada con TTL (Time-To-Live)
class TTLCache:
    """Caché con tiempo de vida para los resultados de reranking."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Inicializa la caché con un tamaño máximo y un tiempo de vida.
        
        Args:
            max_size: Tamaño máximo de la caché
            ttl: Tiempo de vida en segundos (por defecto 1 hora)
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor de la caché si existe y no ha expirado."""
        self._cleanup_expired()
        if key in self.cache:
            # Actualizar timestamp al acceder para implementar LRU
            self.timestamps[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Establece un valor en la caché con el timestamp actual."""
        self._cleanup_expired()
        
        # Si la caché está llena, eliminar la entrada más antigua
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
            self._remove(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Elimina una entrada de la caché."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def _cleanup_expired(self) -> None:
        """Elimina entradas expiradas de la caché."""
        current_time = time.time()
        expired_keys = [k for k, ts in self.timestamps.items() if current_time - ts > self.ttl]
        for key in expired_keys:
            self._remove(key)
    
    def clear(self) -> None:
        """Limpia toda la caché."""
        self.cache.clear()
        self.timestamps.clear()

# Inicializar la caché con TTL de 1 hora
reranking_cache = TTLCache(max_size=100, ttl=3600)

# Mejorar la función de generación de clave de caché
def generate_cache_key(query: str, chunks: List[str]) -> str:
    """
    Genera una clave de caché única y eficiente basada en la consulta y los chunks.
    
    Mejora:
    - Usa un enfoque más compacto con consistencia criptográfica
    - Considera solo muestras de chunks para claves muy largas
    """
    # Normalizar la consulta (minúsculas, eliminar espacios extra)
    query = re.sub(r'\s+', ' ', query.lower().strip())
    
    # Para chunks muy extensos, usar solo muestras para no hacer la clave demasiado larga
    sample_content = ""
    if len(chunks) > 10:
        # Tomar el primer chunk, un chunk del medio y el último chunk
        sample_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
        for chunk in sample_chunks:
            # Tomar los primeros 100 caracteres, los 100 del medio y los últimos 100
            if len(chunk) > 300:
                chunk_sample = chunk[:100] + chunk[len(chunk)//2-50:len(chunk)//2+50] + chunk[-100:]
            else:
                chunk_sample = chunk
            sample_content += chunk_sample
    else:
        # Para pocos chunks, considerar todos pero con muestras
        for chunk in chunks:
            sample_content += chunk[:200]
    
    # Concatenar consulta y muestras de chunks para formar la huella digital
    content = f"{query}|{len(chunks)}|{sample_content}"
    return hashlib.sha256(content.encode()).hexdigest()

# Asegurarse de que NLTK tenga los recursos necesarios
def ensure_nltk_resources():
    """Asegura que los recursos de NLTK necesarios estén disponibles."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt descargado correctamente")
        except Exception as e:
            logger.error(f"Error al descargar recursos NLTK: {e}")
            raise

# Dividir el texto en segmentos más manejables para la evaluación
def split_chunk_for_evaluation(chunk: str, max_length: int = 800) -> str:
    """
    Divide un chunk largo en segmentos más manejables para evaluación LLM.
    
    Args:
        chunk: Texto completo del chunk
        max_length: Longitud máxima del segmento para evaluación
    
    Returns:
        str: Segmento representativo del chunk para evaluación
    """
    # Extraer título si existe
    title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
    title = title_match.group(1) if title_match else ""
    
    # Si el chunk es corto, devolverlo completo
    if len(chunk) <= max_length:
        return chunk
    
    # Estrategia para chunks largos: título + inicio + medio + final
    third = max_length // 3
    intro = chunk[:third]
    
    # Extraer una sección del medio que sea coherente (párrafo completo)
    mid_start = len(chunk) // 2 - third // 2
    mid_end = mid_start + third
    middle = chunk[mid_start:mid_end]
    
    # Extraer el final
    ending = chunk[-third:]
    
    # Construir un segmento representativo
    if title:
        representative = f"# {title}\n\n{intro}...\n\n{middle}...\n\n{ending}"
    else:
        representative = f"{intro}...\n\n{middle}...\n\n{ending}"
    
    return representative

async def optimized_prepare_chunk_data(
    chunks: List[str], 
    openai_client,
    embedding_model: str = "text-embedding-3-small"
) -> Tuple[List[List[str]], List[Any]]:
    """
    Prepara los datos necesarios para el reranking híbrido con optimizaciones:
    - Procesamiento paralelo de tokens
    - Manejo de errores mejorado
    - Control de tasa adaptativo
    
    Args:
        chunks: Lista de chunks a procesar
        openai_client: Cliente de OpenAI
        embedding_model: Modelo de embeddings a utilizar
    
    Returns:
        Tuple con tokens de chunks y embeddings
    """
    # Asegurarse de tener recursos NLTK
    ensure_nltk_resources()
    
    # Generar tokens para cada chunk en paralelo usando ThreadPoolExecutor
    def tokenize_chunk(chunk):
        return word_tokenize(chunk.lower())
    
    with ThreadPoolExecutor() as executor:
        chunk_tokens = list(executor.map(tokenize_chunk, chunks))
    
    # Optimizar tamaño de lote según el número de chunks
    if len(chunks) <= 10:
        BATCH_SIZE = len(chunks)  # Procesar todos a la vez si son pocos
    else:
        BATCH_SIZE = min(16, len(chunks) // 2)  # Ajuste dinámico
    
    chunk_embeddings = []
    retry_count = 0
    max_retries = 3
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_size = len(batch)
        logger.info(f"Procesando lote de embeddings {i//BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1)//BATCH_SIZE} [tamaño: {batch_size}]")
        
        success = False
        while not success and retry_count < max_retries:
            try:
                response = await openai_client.embeddings.create(
                    model=embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                chunk_embeddings.extend(batch_embeddings)
                success = True
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error al procesar lote (intento {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    logger.error(f"Demasiados reintentos, procesando uno por uno")
                    # Fallback: procesar uno por uno
                    for chunk in batch:
                        try:
                            single_response = await openai_client.embeddings.create(
                                model=embedding_model,
                                input=chunk
                            )
                            chunk_embeddings.append(single_response.data[0].embedding)
                        except Exception as inner_e:
                            logger.error(f"Error al procesar embedding individual: {inner_e}")
                            # Insertar un vector de ceros como fallback
                            chunk_embeddings.append([0.0] * 1536)
                    success = True
                else:
                    # Esperar antes de reintentar con backoff exponencial
                    wait_time = 2 ** retry_count
                    logger.info(f"Esperando {wait_time}s antes de reintentar...")
                    await asyncio.sleep(wait_time)
                    # Reducir el tamaño del lote como estrategia adicional
                    BATCH_SIZE = max(1, BATCH_SIZE // 2)
    
    return chunk_tokens, chunk_embeddings

async def improved_evaluate_chunk_relevance(
    ctx: RunContext[Any], 
    query: str, 
    chunk: str,
    evaluation_model: str = "gpt-3.5-turbo-0125"
) -> Tuple[float, Dict[str, float]]:
    """
    Versión mejorada de la evaluación de relevancia que:
    - Devuelve criterios individuales además del puntaje global
    - Utiliza un enfoque estructurado para la extracción de puntajes
    - Maneja errores de forma más robusta
    
    Args:
        ctx: Contexto de ejecución
        query: Consulta del usuario
        chunk: Fragmento a evaluar
        evaluation_model: Modelo LLM para evaluación
    
    Returns:
        Tuple con puntaje global y diccionario de criterios individuales
    """
    # Extraer título para el logging
    title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
    title = title_match.group(1) if title_match else "Chunk sin título"
    
    # Preparar un segmento representativo para evaluación
    truncated_chunk = split_chunk_for_evaluation(chunk)
    
    eval_prompt = f"""
Eres un experto en análisis de documentos normativos. Evalúa la relevancia del siguiente fragmento respecto a la consulta, considerando que se trata de texto regulatorio.

CRITERIOS DE EVALUACIÓN:
1. Pertinencia temática (0-10): ¿El fragmento trata específicamente el tema consultado?
2. Aplicabilidad directa (0-10): ¿Las disposiciones son directamente aplicables al caso planteado?
3. Completitud normativa (0-10): ¿El fragmento contiene disposiciones completas y no cortadas?
4. Jerarquía normativa (0-10): ¿Cuál es el rango jerárquico de la fuente? (Constitución=10, Ley=8-9, Reglamento=6-7, Resolución=4-5, Circular=1-3)
5. Referencias cruzadas (0-10): ¿El fragmento incluye remisiones útiles a otros artículos o normas relevantes para completar la respuesta?

ESCALA DE PUNTUACIÓN:
- 0: Completamente irrelevante/ausente
- 1-3: Mínimamente presente/útil
- 4-6: Moderadamente presente/útil
- 7-8: Altamente presente/útil
- 9-10: Óptimo/Perfecto para el criterio

Consulta: "{query}"

Fragmento a evaluar:
---
{truncated_chunk}
---

INSTRUCCIONES ESPECIALES:
- Para jerarquía normativa: Considera que normas de mayor rango tienen precedencia interpretativa
- Para referencias cruzadas: Valora positivamente fragmentos que incluyan "véase también", "en concordancia con", remisiones a otros artículos, etc.
- Evalúa si el fragmento está completo o cortado artificialmente
- Penaliza fragmentos que requieran contexto adicional para ser útiles

Puntaje global: 0.35*(Pertinencia) + 0.25*(Aplicabilidad) + 0.15*(Completitud) + 0.15*(Jerarquía) + 0.10*(Referencias)

Responde ÚNICAMENTE en formato JSON:
{{
  "pertenencia": valor,
  "aplicabilidad": valor,
  "completitud": valor,
  "jerarquia": valor,
  "referencias": valor,
  "global": valor,
  "justificacion_breve": "explicación en 1-2 líneas sobre los criterios más determinantes"
}}
"""
    try:
        response = await ctx.deps.openai_client.chat.completions.create(
            model=evaluation_model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            # Intentar parsear directamente el JSON
            scores = json.loads(response_text)
            global_score = float(scores.get("global", 0))
            
            # Extraer puntajes individuales o usar valores por defecto
            criteria_scores = {
                "pertenencia": float(scores.get("pertenencia", 0)),
                "aplicabilidad": float(scores.get("aplicabilidad", 0)),
                "completitud": float(scores.get("completitud", 0)),
                "jerarquia": float(scores.get("jerarquia", 0)),
                "referencias": float(scores.get("referencias", 0))
            }
            
            # Validar el puntaje global
            if not (0 <= global_score <= 10):
                # Recalcular si está fuera de rango
                global_score = 0.35 * criteria_scores["pertenencia"] + 0.25 * criteria_scores["aplicabilidad"] + 0.15 * criteria_scores["completitud"] + 0.15 * criteria_scores["jerarquia"] + 0.10 * criteria_scores["referencias"]
                global_score = max(0, min(10, global_score))  # Asegurar rango 0-10
            
            logger.info(f"Chunk '{title[:30]}...' - Puntuación: Global={global_score:.1f}, P={criteria_scores['pertenencia']:.1f}, A={criteria_scores['aplicabilidad']:.1f}, C={criteria_scores['completitud']:.1f}, J={criteria_scores['jerarquia']:.1f}, R={criteria_scores['referencias']:.1f}")
            return global_score, criteria_scores
            
        except (json.JSONDecodeError, ValueError, KeyError) as json_err:
            # Fallback: Extraer números usando regex si falla el JSON
            logger.warning(f"Error al parsear JSON: {json_err}. Intentando extracción con regex...")
            
            # Intentar encontrar un único valor numérico que podría ser el puntaje global
            match = re.search(r'\b(\d+(\.\d+)?)\b', response_text)
            if match:
                global_score = float(match.group(1))
                global_score = max(0, min(10, global_score))  # Asegurar rango 0-10
                logger.info(f"Chunk '{title[:30]}...' - Puntuación extraída con regex: {global_score}")
                # Valor por defecto para criterios individuales
                return global_score, {"pertenencia": global_score, "aplicabilidad": global_score, "completitud": global_score, "jerarquia": global_score, "referencias": global_score}
            else:
                logger.error(f"No se pudo extraer ningún puntaje numérico de: '{response_text}'")
                return 0.0, {"pertenencia": 0, "aplicabilidad": 0, "completitud": 0, "jerarquia": 0, "referencias": 0}
    
    except Exception as e:
        logger.error(f"Error en evaluate_chunk_relevance para '{title[:30]}...': {e}")
        return 0.0, {"pertenencia": 0, "aplicabilidad": 0, "completitud": 0, "jerarquia": 0, "referencias": 0}

def smart_normalize(scores: List[float], min_threshold: float = 0.1) -> np.ndarray:
    """
    Normalización inteligente que:
    - Evita división por cero
    - Mantiene una separación mínima entre valores
    - Aplica una transformación logarítmica para acentuar diferencias significativas
    
    Args:
        scores: Lista de puntajes a normalizar
        min_threshold: Valor mínimo para mantener separación
    
    Returns:
        Array numpy con valores normalizados
    """
    # Convertir a array de numpy si es una lista
    scores_array = np.array(scores, dtype=float)
    
    # Verificar si el array está vacío
    if len(scores_array) == 0:
        return np.array([])
    
    # Si todos los scores son iguales
    if np.all(scores_array == scores_array[0]):
        # Si son todos ceros, devolver ceros
        if scores_array[0] == 0:
            return np.zeros_like(scores_array)
        # Si son todos positivos iguales, devolver unos
        return np.ones_like(scores_array)
    
    # Encontrar mínimo y máximo
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    range_score = max_score - min_score
    
    # Normalización básica
    if range_score > 0:
        normalized = (scores_array - min_score) / range_score
    else:
        # Caso raro pero posible
        normalized = np.zeros_like(scores_array)
    
    # Aplicar transformación logarítmica para acentuar diferencias
    # Añadir min_threshold para evitar log(0)
    log_normalized = np.log1p(normalized + min_threshold)
    
    # Re-normalizar al rango [0,1]
    min_log = np.min(log_normalized)
    max_log = np.max(log_normalized)
    range_log = max_log - min_log
    
    if range_log > 0:
        return (log_normalized - min_log) / range_log
    else:
        return normalized

async def adaptive_hybrid_rerank(
    ctx: RunContext[Any],
    query: str,
    chunks: List[str],
    chunk_tokens: List[List[str]],
    chunk_embeddings: List[Any],
    weights: Dict[str, float] = {"bm25": 0.3, "cosine": 0.3, "llm": 0.4},
    max_for_llm_eval: int = 15,
    diversify: bool = True
) -> List[str]:
    """
    Versión adaptativa mejorada del reranking híbrido que:
    - Ajusta dinámicamente los pesos según características de la consulta
    - Implementa diversificación de resultados
    - Proporciona métricas detalladas
    - Incluye manejo sofisticado de errores
    
    Args:
        ctx: Contexto de ejecución
        query: Consulta del usuario
        chunks: Lista de chunks a reordenar
        chunk_tokens: Tokens de cada chunk
        chunk_embeddings: Embeddings de cada chunk
        weights: Pesos iniciales para cada señal
        max_for_llm_eval: Máximo de chunks para evaluación LLM
        diversify: Si se debe diversificar los resultados
    
    Returns:
        Lista de chunks reordenados
    """
    start_time = time.time()
    
    try:
        if not chunks:
            logger.warning("No hay chunks para reordenar")
            return []
            
        if len(chunks) <= 1:
            return chunks
        
        # Analizar la consulta para ajustar los pesos
        adjusted_weights = await analyze_and_adjust_weights(ctx, query, weights)
        logger.info(f"Pesos ajustados: {adjusted_weights}")
        
        # 1. Señal BM25 (léxica)
        bm25_model = BM25Okapi(chunk_tokens)
        query_tokens = word_tokenize(query.lower())
        bm25_scores = bm25_model.get_scores(query_tokens)
        
        # 2. Señal de similitud coseno (semántica)
        try:
            query_embedding_response = await ctx.deps.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(query_embedding_response.data[0].embedding)
            
            cosine_scores = []
            for emb in chunk_embeddings:
                emb_arr = np.array(emb)
                dot_product = np.dot(query_embedding, emb_arr)
                norm_query = np.linalg.norm(query_embedding)
                norm_emb = np.linalg.norm(emb_arr)
                
                # Evitar división por cero
                denominator = norm_query * norm_emb
                if denominator == 0:
                    cosine = 0
                else:
                    cosine = dot_product / denominator
                
                cosine_scores.append(cosine)
        except Exception as e:
            logger.error(f"Error al calcular similitud coseno: {e}")
            # Fallback: asignar valores neutros
            cosine_scores = [0.5] * len(chunks)
        
        # 3. Señal LLM (evaluación semántica profunda)
        # Evaluar solo para los primeros max_for_llm_eval chunks
        
        # Paso preliminar: Pre-filtrar chunks para LLM usando BM25 y coseno
        if len(chunks) > max_for_llm_eval:
            # Normalizar scores previos
            bm25_norm = smart_normalize(bm25_scores)
            cosine_norm = smart_normalize(cosine_scores)
            
            # Combinar BM25 y coseno para pre-filtrado
            pre_scores = 0.5 * bm25_norm + 0.5 * cosine_norm
            pre_indices = sorted(range(len(chunks)), key=lambda i: pre_scores[i], reverse=True)
            llm_candidate_indices = pre_indices[:max_for_llm_eval]
        else:
            llm_candidate_indices = list(range(len(chunks)))
        
        # Evaluar con LLM los chunks pre-filtrados
        llm_scores = [0] * len(chunks)  # Inicializar todos con cero
        llm_criteria = [{} for _ in range(len(chunks))]  # Almacenar criterios individuales
        
        # Evaluación en paralelo para mejorar velocidad
        eval_tasks = []
        for i in llm_candidate_indices:
            eval_tasks.append(
                improved_evaluate_chunk_relevance(ctx, query, chunks[i])
            )
        
        # Esperar resultados de evaluación
        eval_results = await asyncio.gather(*eval_tasks)
        
        # Asignar resultados a las posiciones correctas
        for idx, (i, (score, criteria)) in enumerate(zip(llm_candidate_indices, eval_results)):
            llm_scores[i] = score
            llm_criteria[i] = criteria
            
            # Extraer título para el logging (solo para los primeros 5)
            if idx < 5:
                title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunks[i])
                title = title_match.group(1) if title_match else f"Chunk {i+1}"
                logger.info(f"Chunk #{idx+1}: '{title[:50]}...' - LLM score: {score:.2f}")
        
        # Normalizar las señales
        bm25_norm = smart_normalize(bm25_scores)
        cosine_norm = smart_normalize(cosine_scores)
        llm_norm = smart_normalize(llm_scores)
        
        # Combinar las señales con los pesos ajustados
        combined_scores = (
            adjusted_weights["bm25"] * bm25_norm + 
            adjusted_weights["cosine"] * cosine_norm + 
            adjusted_weights["llm"] * llm_norm
        )
        
        # Diversificación de resultados (opcional)
        if diversify and len(chunks) > 3:
            ranked_chunks = []
            available_indices = set(range(len(chunks)))
            
            # Índices ordenados por puntaje
            sorted_indices = sorted(range(len(chunks)), key=lambda i: combined_scores[i], reverse=True)
            
            # Siempre incluir el mejor resultado
            top_idx = sorted_indices[0]
            ranked_chunks.append(chunks[top_idx])
            available_indices.remove(top_idx)
            
            # Para los siguientes, alternar entre mejor puntaje y diversidad
            similarity_threshold = 0.8  # Umbral para considerar similares
            
            while available_indices and len(ranked_chunks) < len(chunks):
                # Obtener el siguiente mejor por puntaje
                for idx in sorted_indices:
                    if idx in available_indices:
                        next_candidate = idx
                        break
                else:
                    # No deberíamos llegar aquí, pero por si acaso
                    break
                
                # Verificar si es suficientemente diferente a los ya incluidos
                candidate_emb = np.array(chunk_embeddings[next_candidate])
                
                # Calcular similitud con los últimos 3 chunks incluidos
                recent_chunks = ranked_chunks[-3:] if len(ranked_chunks) >= 3 else ranked_chunks
                recent_indices = [sorted_indices.index(chunks.index(chunk)) for chunk in recent_chunks]
                
                similarities = []
                for idx in recent_indices:
                    recent_emb = np.array(chunk_embeddings[idx])
                    # Evitar división por cero
                    norm_candidate = np.linalg.norm(candidate_emb)
                    norm_recent = np.linalg.norm(recent_emb)
                    if norm_candidate == 0 or norm_recent == 0:
                        similarities.append(0)
                    else:
                        cos_sim = np.dot(candidate_emb, recent_emb) / (norm_candidate * norm_recent)
                        similarities.append(cos_sim)
                
                # Si es muy similar a chunks recientes, buscar uno más diverso
                if len(similarities) > 0 and max(similarities) > similarity_threshold:
                    # Buscar un chunk diverso entre los siguientes mejores
                    diverse_candidate = None
                    
                    for idx in sorted_indices:
                        if idx not in available_indices:
                            continue
                            
                        diverse_emb = np.array(chunk_embeddings[idx])
                        is_diverse = True
                        
                        for recent_idx in recent_indices:
                            recent_emb = np.array(chunk_embeddings[recent_idx])
                            # Evitar división por cero
                            norm_diverse = np.linalg.norm(diverse_emb)
                            norm_recent = np.linalg.norm(recent_emb)
                            if norm_diverse == 0 or norm_recent == 0:
                                continue
                                
                            cos_sim = np.dot(diverse_emb, recent_emb) / (norm_diverse * norm_recent)
                            if cos_sim > similarity_threshold:
                                is_diverse = False
                                break
                        
                        if is_diverse:
                            diverse_candidate = idx
                            break
                    
                    # Si encontramos un candidato diverso, usarlo
                    if diverse_candidate is not None:
                        next_candidate = diverse_candidate
                
                # Añadir el candidato elegido
                ranked_chunks.append(chunks[next_candidate])
                available_indices.remove(next_candidate)
        else:
            # Ordenamiento simple por puntaje combinado
            ranked_indices = sorted(range(len(chunks)), key=lambda i: combined_scores[i], reverse=True)
            ranked_chunks = [chunks[i] for i in ranked_indices]
        
        # Logging de métricas
        processing_time = time.time() - start_time
        logger.info(f"Reranking completado en {processing_time:.2f}s")
        
        # Mostrar las puntuaciones de los 5 mejores chunks
        top5_indices = sorted(range(len(chunks)), key=lambda i: combined_scores[i], reverse=True)[:5]
        logger.info(f"Top 5 puntuaciones: {[combined_scores[i] for i in top5_indices]}")
        
        return ranked_chunks
        
    except Exception as e:
        logger.error(f"Error crítico en reranking adaptativo: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # En caso de error, devolvemos los chunks originales
        return chunks

async def analyze_and_adjust_weights(
    ctx: RunContext[Any], 
    query: str, 
    default_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Analiza la consulta para ajustar dinámicamente los pesos del reranking
    para un agente experto en legal y compliance genérico.
    
    Características que influyen:
    - Tipo de dominio legal (financiero, laboral, corporativo, etc.)
    - Naturaleza de la consulta (interpretación vs. búsqueda específica)
    - Nivel de especificidad técnica
    - Referencias temporales o jurisdiccionales
    
    Args:
        ctx: Contexto de ejecución
        query: Consulta del usuario
        default_weights: Pesos por defecto
    
    Returns:
        Diccionario con pesos ajustados
    """
    # Copiar pesos por defecto
    weights = default_weights.copy()
    query_lower = query.lower()
    
    # === ANÁLISIS DE DOMINIO LEGAL ===
    
    # Términos técnicos legales generales
    legal_technical_terms = [
        # Documentos legales
        "artículo", "artículos", "inciso", "fracción", "párrafo", "capítulo",
        "ley", "decreto", "reglamento", "resolución", "circular", "acuerdo",
        "código", "constitución", "convenio", "tratado", "norma", "normativa",
        
        # Procesos legales
        "jurisprudencia", "sentencia", "resolución", "dictamen", "criterio",
        "interpretación", "precedente", "caso", "expediente",
        
        # Compliance y regulatorio
        "cumplimiento", "infracción", "sanción", "multa", "penalización",
        "auditoría", "supervisión", "autoridad", "regulador", "organismo",
        "debido proceso", "diligencia debida", "kyc", "aml"
    ]
    
    # Términos por dominio específico
    financial_terms = [
        "financiero", "bancario", "bursátil", "seguros", "fintech",
        "visa", "mastercard", "tarifa", "comisión", "interchange",
        "adquirente", "emisor", "transacción", "lavado", "prevención",
        "cnbv", "banxico", "condusef", "shcp"
    ]
    
    corporate_terms = [
        "corporativo", "empresarial", "mercantil", "societario",
        "consejo", "directorio", "accionista", "dividendo", "fusión",
        "adquisición", "joint venture", "gobierno corporativo"
    ]
    
    labor_terms = [
        "laboral", "trabajo", "empleado", "trabajador", "sindicato",
        "contrato", "despido", "liquidación", "prestaciones",
        "stps", "imss", "infonavit", "fonacot"
    ]
    
    tax_terms = [
        "fiscal", "tributario", "impuesto", "deducción", "isr", "iva",
        "ieps", "sat", "cff", "lisr", "liva", "declaración"
    ]
    
    data_privacy_terms = [
        "privacidad", "protección", "datos", "personales", "inai",
        "avisos", "consentimiento", "transferencia", "arco", "gdpr"
    ]
    
    environmental_terms = [
        "ambiental", "ecológico", "emisiones", "residuos", "impacto",
        "semarnat", "profepa", "asea", "eia", "mia"
    ]
    
    # Detectar dominio específico
    has_financial = any(term in query_lower for term in financial_terms)
    has_corporate = any(term in query_lower for term in corporate_terms)
    has_labor = any(term in query_lower for term in labor_terms)
    has_tax = any(term in query_lower for term in tax_terms)
    has_privacy = any(term in query_lower for term in data_privacy_terms)
    has_environmental = any(term in query_lower for term in environmental_terms)
    has_legal_technical = any(term in query_lower for term in legal_technical_terms)
    
    # === ANÁLISIS DE TIPO DE CONSULTA ===
    
    # Consultas de interpretación (necesitan más análisis semántico)
    interpretation_indicators = [
        "qué significa", "cómo interpretar", "qué implica", "alcance de",
        "criterio", "interpretación", "análisis", "opinión", "considera",
        "aplicable", "aplica", "abarca", "incluye", "comprende"
    ]
    is_interpretation = any(indicator in query_lower for indicator in interpretation_indicators)
    
    # Consultas específicas de artículo/norma (necesitan coincidencia exacta)
    specific_article_indicators = [
        "artículo", "art.", "art ", "inciso", "fracción", "párrafo",
        "capítulo", "cap.", "cap ", "título", "sección", "anexo"
    ]
    seeks_specific_article = any(indicator in query_lower for indicator in specific_article_indicators)
    
    # Referencias jurisdiccionales/geográficas
    jurisdiction_indicators = [
        "federal", "estatal", "local", "municipal", "cdmx", "ciudad de méxico",
        "estados unidos", "méxico", "europeo", "internacional", "nacional"
    ]
    has_jurisdiction = any(indicator in query_lower for indicator in jurisdiction_indicators)
    
    # Referencias temporales
    temporal_indicators = [
        "2024", "2025", "2023", "actual", "vigente", "nuevo", "nueva",
        "reciente", "último", "actualizado", "modificación", "reforma",
        "derogado", "abrogado", "anterior", "previo"
    ]
    has_temporal_reference = any(indicator in query_lower for indicator in temporal_indicators)
    
    # Complejidad de la consulta
    query_length = len(query.split())
    
    # === AJUSTE DE PESOS SEGÚN ANÁLISIS ===
    
    # 1. Consultas que buscan artículo/norma específica
    if seeks_specific_article:
        weights["bm25"] = 0.50  # Alta prioridad a coincidencias exactas
        weights["cosine"] = 0.25
        weights["llm"] = 0.25
    
    # 2. Consultas de interpretación legal
    elif is_interpretation:
        weights["bm25"] = 0.20  # Baja prioridad a coincidencias exactas
        weights["cosine"] = 0.30
        weights["llm"] = 0.50  # Alta prioridad a análisis semántico
    
    # 3. Consultas con dominio específico muy técnico
    elif has_financial or has_tax or has_privacy:
        weights["bm25"] = 0.40  # Términos técnicos exactos importantes
        weights["cosine"] = 0.30
        weights["llm"] = 0.30
    
    # 4. Consultas con términos legales técnicos generales
    elif has_legal_technical:
        weights["bm25"] = 0.35
        weights["cosine"] = 0.35
        weights["llm"] = 0.30
    
    # 5. Consultas muy cortas (pocas palabras)
    elif query_length <= 3:
        weights["bm25"] = 0.25
        weights["cosine"] = 0.30
        weights["llm"] = 0.45  # Necesita más interpretación
    
    # 6. Consultas muy largas y complejas
    elif query_length >= 20:
        weights["bm25"] = 0.20
        weights["cosine"] = 0.25
        weights["llm"] = 0.55  # Necesita comprensión profunda
    
    # === AJUSTES ESPECIALES ===
    
    # Ajuste por referencias temporales
    if has_temporal_reference:
        # Favor a evaluación semántica para contenido reciente
        weights["bm25"] = max(0.15, weights["bm25"] - 0.10)
        weights["llm"] = min(0.60, weights["llm"] + 0.10)
    
    # Ajuste por referencias jurisdiccionales
    if has_jurisdiction:
        # Favor a términos exactos para jurisdicciones específicas
        weights["bm25"] = min(0.50, weights["bm25"] + 0.05)
        weights["cosine"] = max(0.20, weights["cosine"] - 0.05)
    
    # Dominios que requieren precisión extrema (financiero, fiscal)
    if has_financial or has_tax:
        weights["bm25"] = min(0.55, weights["bm25"] + 0.05)
    
    # === NORMALIZACIÓN FINAL ===
    total = sum(weights.values())
    if total != 1.0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights


# === FUNCIÓN AUXILIAR PARA DEBUG ===
def explain_weight_adjustment(query: str, weights: Dict[str, float]) -> str:
    """
    Explica por qué se ajustaron los pesos de cierta manera.
    Útil para debugging y logging.
    """
    query_lower = query.lower()
    explanations = []
    
    # Detectar características de la consulta
    if any(term in query_lower for term in ["artículo", "inciso", "fracción"]):
        explanations.append("Busca artículo específico → Mayor peso a BM25")
    
    if any(term in query_lower for term in ["significa", "interpretar", "criterio"]):
        explanations.append("Consulta de interpretación → Mayor peso a LLM")
    
    if any(term in query_lower for term in ["financiero", "fiscal", "visa", "mastercard"]):
        explanations.append("Dominio técnico específico → Balance BM25/Coseno")
    
    if any(term in query_lower for term in ["2024", "2025", "reciente", "nuevo"]):
        explanations.append("Referencia temporal → Mayor peso a LLM")
    
    if len(query.split()) <= 3:
        explanations.append("Consulta muy corta → Mayor peso a LLM")
    elif len(query.split()) >= 20:
        explanations.append("Consulta muy compleja → Mayor peso a LLM")
    
    explanation = f"Pesos: BM25={weights['bm25']:.2f}, Coseno={weights['cosine']:.2f}, LLM={weights['llm']:.2f}"
    if explanations:
        explanation += f"\nRazones: {'; '.join(explanations)}"
    
    return explanation



async def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Cuenta el número de tokens en un texto utilizando tiktoken.
    
    Args:
        text: Texto a contar
        model: Modelo para el conteo de tokens
    
    Returns:
        int: Número de tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback: aproximación basada en palabras
        return len(text.split()) * 1.3

# Función principal mejorada para reranking
async def rerank_chunks(
    ctx: RunContext[Any], 
    query: str, 
    chunks: List[str], 
    max_to_rerank: int = 15,
    diversify: bool = True,
    use_cache: bool = True
) -> List[str]:
    """
    Función principal mejorada para reordenar chunks según su relevancia.
    
    Características:
    - Caché con TTL para resultados recurrentes
    - Evaluación adaptativa según tipo de consulta
    - Diversificación de resultados
    - Manejo de errores mejorado
    - Métricas detalladas
    
    Args:
        ctx: Contexto de ejecución
        query: Consulta del usuario
        chunks: Lista de chunks a reordenar
        max_to_rerank: Máximo de chunks para evaluación LLM completa
        diversify: Si se debe diversificar los resultados
        use_cache: Si se debe utilizar la caché
    
    Returns:
        Lista de chunks reordenados
    """
    logger.info(f"Iniciando reranking para {len(chunks)} chunks (max LLM: {max_to_rerank}, diversificar: {diversify})")
    start_time = time.time()
    
    if not chunks:
        logger.warning("No hay chunks para reordenar")
        return []
        
    if len(chunks) == 1:
        return chunks
    
    # Generar clave de caché
    if use_cache:
        cache_key = generate_cache_key(query, chunks)
        
        # Verificar si ya tenemos estos resultados en caché
        cached_result = reranking_cache.get(cache_key)
        if cached_result:
            logger.info("Resultados encontrados en caché, omitiendo procesamiento de reranking")
            return cached_result
    
    try:
        # Preparamos los datos necesarios para el reranking híbrido
        logger.info("Preparando datos para reranking adaptativo")
        chunk_tokens, chunk_embeddings = await optimized_prepare_chunk_data(
            chunks, 
            ctx.deps.openai_client
        )
        
        # Ejecutar reranking adaptativo
        ranked_chunks = await adaptive_hybrid_rerank(
            ctx, 
            query, 
            chunks, 
            chunk_tokens, 
            chunk_embeddings,
            max_for_llm_eval=max_to_rerank,
            diversify=diversify
        )
        
        # Guardar resultados en caché
        if use_cache:
            reranking_cache.set(cache_key, ranked_chunks)
        
        # Métricas de rendimiento
        processing_time = time.time() - start_time
        logger.info(f"Reranking completado en {processing_time:.2f}s")
        
        return ranked_chunks
        
    except Exception as e:
        logger.error(f"Error en reranking adaptativo: {e}")
        logger.info("Fallback: Ejecutando reranking simple")
        
        try:
            # Método fallback: reranking básico con LLM
            fallback_results = await rerank_chunks_with_llm(ctx, query, chunks, max_to_rerank)
            return fallback_results
        except Exception as fallback_error:
            logger.error(f"Fallback también falló: {fallback_error}")
            logger.info("Devolviendo chunks sin reordenar")
            return chunks













# Mantener la función legada para compatibilidad
async def rerank_chunks_with_llm(ctx: RunContext[Any], query: str, chunks: List[str], max_to_rerank: int = 15) -> List[str]:
    """
    Versión simplificada del reranking que usa solo LLM para evaluación.
    Se mantiene como fallback y para compatibilidad.
    """
    logger.info(f"Iniciando reranking simple con LLM para {len(chunks)} chunks (máximo: {max_to_rerank})")
    
    if len(chunks) <= 1:
        return chunks
        
    chunks_to_rerank = chunks[:max_to_rerank]
    remaining_chunks = chunks[max_to_rerank:] if len(chunks) > max_to_rerank else []
    
    ranked_chunks = []
    for i, chunk in enumerate(chunks_to_rerank):
        # Extraer título para el logging
        title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
        title = title_match.group(1) if title_match else f"Chunk {i+1}"
        
        # Preparar un segmento representativo
        representative = split_chunk_for_evaluation(chunk)
        
        eval_prompt = f"""
Evalúa la relevancia del siguiente fragmento respecto a la consulta:

Consulta: "{query}"

Fragmento:
---
{representative}
---

Asigna un puntaje de 0 a 10, donde 10 es extremadamente relevante.
Responde solo con el número.
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
                # Intentar extraer un número
                match = re.search(r'\d+(\.\d+)?', score_text)
                if match:
                    score = float(match.group(0))
                else:
                    # Si no hay coincidencia, intentar convertir directamente
                    score = float(score_text)
                    
                logger.info(f"Chunk '{title[:50]}...' recibió puntuación: {score}")
                ranked_chunks.append((chunk, score))
            except (ValueError, TypeError):
                logger.warning(f"No se pudo extraer puntaje numérico de '{score_text}', asignando 0")
                ranked_chunks.append((chunk, 0))
        except Exception as e:
            logger.error(f"Error evaluando chunk {i+1}: {e}")
            ranked_chunks.append((chunk, 0))
    
    # Ordenar por puntuación
    ranked_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Logging de puntuaciones
    if ranked_chunks:
        logger.info(f"Reranking simple completado. Top puntuaciones: {[score for _, score in ranked_chunks[:5]]}")
    
    # Devolver chunks ordenados más los restantes sin cambios
    return [chunk for chunk, _ in ranked_chunks] + remaining_chunks