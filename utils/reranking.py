"""
Módulo para reranking de resultados RAG utilizando LLM.
"""

import re
import logging
from typing import List, Any
from pydantic_ai import RunContext

# Configuración de logging
logger = logging.getLogger(__name__)


async def rerank_chunks_with_llm(ctx: RunContext[Any], query: str, chunks: list, max_to_rerank: int = 15) -> list:
    """
    Reordena los chunks recuperados utilizando un LLM para evaluar su relevancia con respecto a la query.
    
    Args:
        ctx: Contexto de ejecución con dependencias
        query: Consulta del usuario
        chunks: Lista de chunks a reordenar (cada chunk es un texto)
        max_to_rerank: Número máximo de chunks a reordenar (para controlar costos)
        
    Returns:
        Lista de chunks reordenados por relevancia
    """
    logger.info(f"Iniciando reranking con LLM para {len(chunks)} chunks (máximo a reordenar: {max_to_rerank})")
    
    # Si hay pocos chunks, no hace falta reordenar
    if len(chunks) <= 1:
        return chunks
    
    # Limitar la cantidad de chunks a reordenar para controlar costos
    chunks_to_rerank = chunks[:max_to_rerank]
    remaining_chunks = chunks[max_to_rerank:] if len(chunks) > max_to_rerank else []
    
    # Preparamos los pares (chunk, score) para almacenar los resultados
    ranked_chunks = []
    
    try:
        # Evaluamos cada chunk de forma individual
        for i, chunk in enumerate(chunks_to_rerank):
            # Extraer título del chunk (asumiendo formato específico)
            title_match = re.search(r'#\s+(.+?)(?:\n|\[|$)', chunk)
            title = title_match.group(1) if title_match else f"Chunk {i+1}"
            
            # Crear un prompt para evaluar la relevancia
            eval_prompt = f"""
Necesito que evalúes la relevancia del siguiente fragmento de texto respecto a una consulta.

Consulta: "{query}"

Fragmento a evaluar:
---
{chunk[:1500]}
---

Basándote únicamente en la relevancia directa para responder a la consulta, asigna una puntuación del 0 al 10:
- 10: Extremadamente relevante, contiene exactamente la información solicitada
- 7-9: Muy relevante, contiene información importante relacionada con la consulta
- 4-6: Moderadamente relevante, tiene alguna relación con la consulta
- 1-3: Poco relevante, menciona conceptos relacionados pero no es útil
- 0: Irrelevante, no ayuda a responder la consulta

Responde ÚNICAMENTE con el número de la puntuación, sin explicaciones adicionales.
"""
            try:
                # Llamada al LLM para evaluar
                response = await ctx.deps.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",  # Modelo más económico para reranking
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.1,  # Baja temperatura para decisiones más consistentes
                    max_tokens=5  # Solo necesitamos un número
                )
                
                # Extraer el score del texto de respuesta
                score_text = response.choices[0].message.content.strip()
                
                # Intentar convertir a número
                try:
                    score = float(re.search(r'\d+(\.\d+)?', score_text).group(0))
                    logger.debug(f"Chunk '{title}' recibió puntuación: {score}")
                    ranked_chunks.append((chunk, score))
                except (ValueError, AttributeError):
                    logger.warning(f"No se pudo extraer puntuación numérica de '{score_text}', asignando 0")
                    ranked_chunks.append((chunk, 0))
                    
            except Exception as e:
                logger.error(f"Error evaluando chunk {i+1}: {e}")
                # Si hay un error, asignamos puntuación 0 para no perder el chunk
                ranked_chunks.append((chunk, 0))
        
        # Ordenar por puntuación (de mayor a menor)
        ranked_chunks.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Reranking completado. Top 3 puntuaciones: {[score for _, score in ranked_chunks[:3]]}")
        
        # Retornar solo los textos, ya ordenados
        reranked_chunks = [chunk for chunk, _ in ranked_chunks]
        
        # Añadir los chunks que no fueron reordenados al final
        return reranked_chunks + remaining_chunks
        
    except Exception as e:
        logger.error(f"Error en el proceso de reranking: {e}")
        # En caso de error, devolver los chunks en su orden original
        return chunks