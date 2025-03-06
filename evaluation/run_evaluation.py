#!/usr/bin/env python
"""
Script para ejecutar la evaluación completa del sistema RAG
"""
import argparse
import asyncio
import logging
import os
import sys
import random  # Añadida importación para random.sample
from typing import List, Dict, Any, Optional
import json
import pandas as pd

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz del proyecto al PYTHONPATH
# Esto permite importar módulos desde la raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# Importar los módulos de evaluación
try:
    # Importar desde el mismo directorio
    from evaluation.ragas_evaluation import RAGEvaluator
    from evaluation.component_evaluation import ComponentEvaluator
    logger.info("Módulos importados usando importación absoluta")
except ImportError:
    # Alternativa si lo anterior falla
    try:
        from .ragas_evaluation import RAGEvaluator
        from .component_evaluation import ComponentEvaluator
        logger.info("Módulos importados usando importación relativa")
    except ImportError as e:
        logger.error(f"No se pudieron importar los módulos de evaluación: {e}")
        logger.error("Asegúrate de que los archivos están en el mismo directorio.")
        logger.error("Si el error persiste, ejecuta el script desde la raíz del proyecto con:")
        logger.error("python -m evaluation.run_evaluation [comando] [opciones]")
        sys.exit(1)

# Importar módulos de agentes
from agents.ai_expert_v1 import ai_expert, AIDeps
from agents.orchestrator_agent import process_query, OrchestratorDeps, OrchestrationResult

async def run_full_evaluation(
    json_path: str,
    output_dir: str = "evaluation_results",
    sample_size: Optional[int] = None,
    use_orchestrator: bool = True
):
    """
    Ejecuta la evaluación completa del sistema RAG utilizando ambos módulos.
    
    Args:
        json_path: Ruta al archivo JSON con el testset
        output_dir: Directorio para guardar los resultados
        sample_size: Tamaño de la muestra para evaluación (None para usar todo el dataset)
        use_orchestrator: Si usar el orquestador o no
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener variables de entorno
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([supabase_url, supabase_key, openai_api_key]):
        logger.error("Faltan variables de entorno necesarias. Asegúrate de configurar SUPABASE_URL, SUPABASE_KEY y OPENAI_API_KEY.")
        return
    
    logger.info(f"Iniciando evaluación RAG completa con el archivo {json_path}")
    
    # Parte 1: Evaluación con RAGAS
    try:
        logger.info("Iniciando evaluación con RAGAS...")
        ragas_evaluator = RAGEvaluator(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            openai_api_key=openai_api_key
        )
        
        # Cargar dataset
        ragas_evaluator.load_dataset_from_json(json_path)
        
        # Evaluar sistema
        metrics = await ragas_evaluator.evaluate_rag_system(
            use_orchestrator=use_orchestrator,
            output_path=os.path.join(output_dir, "ragas_results.csv")
        )
        
        # Generar informe
        ragas_evaluator.generate_evaluation_report(
            metrics, 
            output_path=os.path.join(output_dir, "ragas_report.md")
        )
        
        logger.info("Evaluación RAGAS completada con éxito")
    except Exception as e:
        logger.error(f"Error en la evaluación RAGAS: {e}")
    
    # Parte 2: Evaluación de componentes
    try:
        logger.info("Iniciando evaluación de componentes individuales...")
        component_evaluator = ComponentEvaluator(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            openai_api_key=openai_api_key
        )
        
        # Cargar dataset
        component_evaluator.load_dataset_from_json(json_path)
        
        # Evaluar componentes
        retrieval_results = await component_evaluator.evaluate_retrieval_component(num_samples=sample_size)
        generation_results = await component_evaluator.evaluate_generation_component(num_samples=sample_size)
        
        # Guardar resultados en CSV
        retrieval_results.to_csv(os.path.join(output_dir, "retrieval_evaluation.csv"), index=False)
        generation_results.to_csv(os.path.join(output_dir, "generation_evaluation.csv"), index=False)
        
        # Visualizar resultados
        component_evaluator.visualize_results(
            retrieval_results, 
            generation_results,
            output_path=os.path.join(output_dir, "component_evaluation.png")
        )
        
        # Generar informe detallado
        component_evaluator.generate_detailed_report(
            retrieval_results, 
            generation_results,
            output_path=os.path.join(output_dir, "detailed_evaluation_report.md")
        )
        
        logger.info("Evaluación de componentes completada con éxito")
    except Exception as e:
        logger.error(f"Error en la evaluación de componentes: {e}")
    
    logger.info(f"Evaluación completa finalizada. Resultados guardados en el directorio: {output_dir}")

def prepare_testset(input_json: str, output_json: str, num_samples: Optional[int] = None):
    """
    Prepara un conjunto de pruebas a partir del JSON original.
    Puede filtrar, limpiar o seleccionar un subconjunto aleatorio.
    
    Args:
        input_json: Ruta al archivo JSON original
        output_json: Ruta donde guardar el testset preparado
        num_samples: Número de muestras a seleccionar (None para usar todas)
    """
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convertir a lista de elementos si es un diccionario
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data
        
        # Seleccionar muestra aleatoria si se especifica
        if num_samples and num_samples < len(items):
            selected_items = random.sample(items, num_samples)
        else:
            selected_items = items
        
        # Convertir de nuevo a diccionario
        prepared_data = {str(i): item for i, item in enumerate(selected_items, 1)}
        
        # Guardar el testset preparado
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(prepared_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Testset preparado con {len(prepared_data)} ejemplos guardado en {output_json}")
        return True
    except Exception as e:
        logger.error(f"Error al preparar el testset: {e}")
        return False

def main():
    """Función principal que maneja los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Herramienta de evaluación para sistemas RAG")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subcomando para preparar dataset
    prepare_parser = subparsers.add_parser("prepare", help="Preparar dataset para evaluación")
    prepare_parser.add_argument("--input", required=True, help="Ruta al archivo JSON de entrada")
    prepare_parser.add_argument("--output", required=True, help="Ruta para guardar el dataset procesado")
    prepare_parser.add_argument("--samples", type=int, help="Número de muestras a incluir (opcional)")
    
    # Subcomando para evaluación RAGAS
    ragas_parser = subparsers.add_parser("ragas", help="Ejecutar evaluación RAGAS")
    ragas_parser.add_argument("--dataset", required=True, help="Ruta al dataset de evaluación")
    ragas_parser.add_argument("--output", required=True, help="Directorio para guardar los resultados")
    ragas_parser.add_argument("--samples", type=int, help="Número de muestras a evaluar (opcional)")
    ragas_parser.add_argument("--no-orchestrator", action="store_true", 
                           help="No usar el orquestador (usar directamente AI Expert)")
    
    # Subcomando para evaluación de componentes
    component_parser = subparsers.add_parser("component", help="Ejecutar evaluación de componentes")
    component_parser.add_argument("--dataset", required=True, help="Ruta al dataset de evaluación")
    component_parser.add_argument("--output", required=True, help="Directorio para guardar los resultados")
    component_parser.add_argument("--samples", type=int, help="Número de muestras a evaluar (opcional)")
    
    args = parser.parse_args()
    
    # Ejecutar el comando correspondiente
    if args.command == "prepare":
        prepare_testset(args.input, args.output, args.samples)
    elif args.command == "ragas":
        asyncio.run(run_full_evaluation(
            args.dataset,
            args.output,
            args.samples,
            not args.no_orchestrator
        ))
    elif args.command == "component":
        asyncio.run(run_full_evaluation(
            args.dataset,
            args.output,
            args.samples,
            False
        ))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()