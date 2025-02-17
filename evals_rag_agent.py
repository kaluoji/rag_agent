import asyncio
import json
import logging
from statistics import mean
from pathlib import Path
from typing import List, Dict, Any

# Importa el agente y dependencias
from agents.ai_expert_v1 import ai_expert, AIDeps
from utils.supabase_wrapper import SupabaseClientWrapper  # Nueva importación
from tests.test_rag_agent import FakeAIDeps, FakeOpenAIClient
from pydantic_ai import models


models.ALLOW_MODEL_REQUESTS = True

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_response(response_text: str, expected_keywords: List[str]) -> int:
    """
    Evalúa la respuesta en función de las palabras clave, estructura y nivel de detalle.

    Puntuación:
      - +5 por cada palabra clave encontrada (sin distinguir mayúsculas/minúsculas).
      - +3 bonus si se detecta estructura (listas numeradas, viñetas o múltiples saltos de línea).
      - -2 por cada frase genérica que indique falta de detalle.
      - -5 si la respuesta tiene menos de 50 tokens.
    
    Args:
        response_text (str): Texto de la respuesta.
        expected_keywords (List[str]): Lista de palabras clave esperadas.
    
    Returns:
        int: Puntuación calculada.
    """
    score = 0
    response_lower = response_text.lower()
    
    # Puntos por palabras clave
    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            score += 5

    # Bonus por estructura (listados, viñetas o saltos de línea dobles)
    structure_markers = ["1.", "2.", "3.", "-", "*", "\n\n"]
    if any(marker in response_text for marker in structure_markers):
        score += 3

    # Penalización por frases genéricas que indiquen falta de detalle
    generic_phrases = ["proporcione más detalles", "necesito más información", "no encontré", "no se encontró"]
    for phrase in generic_phrases:
        if phrase in response_lower:
            score -= 2

    # Penalización si la respuesta es demasiado corta
    token_count = len(response_text.split())
    if token_count < 50:
        score -= 5

    return score

async def process_case(case: Dict[str, Any], deps: FakeAIDeps) -> Dict[str, Any]:
    """
    Procesa un caso de evaluación individual.

    Args:
        case (Dict[str, Any]): Diccionario con 'query' y 'expected_keywords'.
        deps (FakeAIDeps): Dependencias necesarias para ejecutar el agente.
    
    Returns:
        Dict[str, Any]: Resultado con la consulta, puntuación y respuesta.
    """
    query = case.get("query", "")
    expected_keywords = case.get("expected_keywords", [])
    try:
        response = await ai_expert.run(query, deps=deps)
    except Exception as e:
        logger.error(f"Error al procesar la consulta '{query}': {e}")
        return {"query": query, "score": None, "response": "", "error": str(e)}

    score = evaluate_response(response.data, expected_keywords)
    return {"query": query, "score": score, "response": response.data}

async def run_evals():
    """
    Carga el archivo JSON de evaluación, procesa cada caso de forma concurrente y 
    muestra el resultado individual y global.
    """
    eval_file = Path("evals_cases.json")
    if not eval_file.exists():
        logger.error("No se encontró el archivo evals_cases.json")
        return

    try:
        with eval_file.open("r", encoding="utf-8") as f:
            eval_cases = json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar el archivo JSON: {e}")
        return

    # Usamos FakeAIDeps en lugar de AIDeps
    deps = FakeAIDeps(
        supabase=SupabaseClientWrapper(is_test=True).get_client(),
        openai_client=FakeOpenAIClient()
    )

    # Procesa los casos de evaluación de forma concurrente
    tasks = [process_case(case, deps) for case in eval_cases]
    results = await asyncio.gather(*tasks)

    scores = []
    for result in results:
        query = result.get("query")
        score = result.get("score")
        response_data = result.get("response")
        if score is not None:
            scores.append(score)
        logger.info(f"Consulta: {query}\nScore: {score}\nRespuesta: {response_data}\n{'-'*40}")

    overall_score = mean(scores) if scores else 0
    logger.info(f"Score global de evaluación: {overall_score:.2f}")

if __name__ == "__main__":
    asyncio.run(run_evals())
