import os
import json
import asyncio
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from dotenv import load_dotenv
import traceback
from ragas.run_config import RunConfig



my_run_config = RunConfig(max_workers=4, timeout=60)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Importar la configuración desde config/config.py
from config.config import settings

# Importa tu Agente RAG y las dependencias definidas en tu proyecto.
from agents.ai_expert_v1 import ai_expert, AIDeps, retrieve_relevant_documentation
from pydantic_ai import RunContext
from supabase import create_client
from openai import AsyncOpenAI

async def main():
    # Cargar el dataset de evaluación desde evaluation/data/testset.json
    dataset_path = os.path.join("evaluation", "data", "testset.json")
    logger.info(f"Cargando dataset desde: {dataset_path}")
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            testset_data = json.load(f)
        
        # Convertir testset_data a una lista de muestras:
        if isinstance(testset_data, dict):
            samples = list(testset_data.values())
        elif isinstance(testset_data, list):
            samples = testset_data
        else:
            raise ValueError("Formato de testset desconocido")
        
        logger.info(f"Número de muestras: {len(samples)}")
        if samples:
            logger.info(f"Primer muestra: {samples[0]}")
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Inicializar las dependencias usando la configuración
    logger.info("Inicializando clientes Supabase y OpenAI")
    try:
        supabase_client = create_client(settings.supabase_url, settings.supabase_key)
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        deps = AIDeps(supabase=supabase_client, openai_client=openai_client)
        logger.info("Clientes inicializados correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar dependencias: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Listas para almacenar los datos de evaluación
    eval_questions = []
    static_contexts = []   # Contexto estático (proporcionado en el JSON)
    dynamic_contexts = []  # Contexto obtenido dinámicamente
    eval_answers = []
    eval_references = []
    
    # Iterar sobre cada muestra
    for idx, sample in enumerate(samples):
        logger.info(f"Procesando muestra {idx+1}/{len(samples)}")
        
        try:
            if isinstance(sample, dict):
                # Soporta tanto "question"/"reference"/"context" como "pregunta"/"respuesta"/"contexto"
                question = sample.get("question", sample.get("question", ""))
                reference = sample.get("reference", sample.get("answer", ""))
                static_ctx = sample.get("context", sample.get("context", ""))
            elif isinstance(sample, str):
                question = sample
                reference = ""
                static_ctx = ""
            else:
                logger.warning(f"Tipo de muestra no soportado: {type(sample)}")
                continue
            
            logger.info(f"Pregunta: {question}")
            
            # Ejecutar el agente RAG para obtener la respuesta generada
            logger.info("Ejecutando agente RAG para obtener respuesta")
            response = await ai_expert.run(question, deps=deps)
            
            # Extraer la respuesta
            if hasattr(response, "data"):
                answer = response.data
            else:
                answer = str(response)
            
            logger.info(f"Respuesta generada (primeros 100 caracteres): {answer[:100]}...")
            
            # Obtener el contexto dinámico
            dynamic_ctx = ""
            logger.info("Intentando obtener el contexto dinámico")
            
            # Método 1: Obtener de los metadatos de la respuesta si existen
            if hasattr(response, "metadata") and response.metadata and "context" in response.metadata:
                dynamic_ctx = response.metadata["context"]
                logger.info("Contexto obtenido de los metadatos de la respuesta")
            else:
                logger.info("No se encontró contexto en los metadatos, intentando alternativas")
                
                # Método 2: Obtener a través de la herramienta directamente
                try:
                    # Crear un contexto de ejecución para la herramienta
                    ctx = RunContext(
                        model=ai_expert.model,
                        prompt=ai_expert.system_prompt,
                        deps=deps,
                        usage=None
                    )
                    
                    # Intento 1: Acceder a través del diccionario de herramientas
                    logger.info("Intentando acceder a la herramienta a través del diccionario de herramientas")
                    if hasattr(ai_expert, "tools") and "retrieve_relevant_documentation" in ai_expert.tools:
                        dynamic_ctx = await ai_expert.tools["retrieve_relevant_documentation"].function(ctx, question)
                        logger.info("Contexto obtenido a través del diccionario de herramientas")
                    
                    # Intento 2: Llamar directamente a la función importada
                    if not dynamic_ctx or dynamic_ctx == "No relevant documentation found.":
                        logger.info("Intentando llamar directamente a la función importada")
                        dynamic_ctx = await retrieve_relevant_documentation(ctx, question)
                        logger.info("Contexto obtenido a través de la función importada")
                    
                    # Verificar si se obtuvo contexto
                    if not dynamic_ctx or dynamic_ctx == "No relevant documentation found.":
                        logger.warning(f"No se encontró contexto relevante para la pregunta: {question}")
                except Exception as e:
                    logger.error(f"Error al obtener el contexto dinámico: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Log del tamaño del contexto dinámico
            if dynamic_ctx:
                logger.info(f"Longitud del contexto dinámico: {len(dynamic_ctx)} caracteres")
            else:
                logger.warning("No se obtuvo ningún contexto dinámico")
            
            # Almacenar los datos en las listas correspondientes
            eval_questions.append(question)
            static_contexts.append([static_ctx])  # Se envuelve en lista, ya que RAGAS espera una lista de contextos
            dynamic_contexts.append([dynamic_ctx])
            eval_answers.append(answer)
            eval_references.append(reference)
            
            logger.info(f"Contexto estático: {static_ctx[:150]}..." if len(static_ctx) > 150 else f"Contexto estático: {static_ctx}")
            logger.info(f"Contexto dinámico: {dynamic_ctx[:150]}..." if len(dynamic_ctx) > 150 else f"Contexto dinámico: {dynamic_ctx}")
            logger.info("----------")
        
        except Exception as e:
            logger.error(f"Error procesando la muestra {idx+1}: {str(e)}")
            logger.error(traceback.format_exc())
            # Continuar con la siguiente muestra
            continue
    
    # Verificar si hay datos suficientes para la evaluación
    if not eval_questions:
        logger.error("No hay datos para evaluar. Todas las muestras fallaron.")
        return
    
    logger.info(f"Total de preguntas procesadas: {len(eval_questions)}")
    
    # Construir dos datasets de evaluación: uno con el contexto estático y otro con el dinámico.
    try:
        logger.info("Construyendo datasets para evaluación")
        
        static_dataset = Dataset.from_dict({
            "question": eval_questions,
            "contexts": static_contexts,
            "answer": eval_answers,
            "reference": eval_references,
        })
        
        dynamic_dataset = Dataset.from_dict({
            "question": eval_questions,
            "contexts": dynamic_contexts,
            "answer": eval_answers,
            "reference": eval_references,
        })
        
        logger.info("Datasets construidos correctamente")
    except Exception as e:
        logger.error(f"Error al construir los datasets: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Evaluar ambos datasets usando los métricos de RAGAS
    try:
        logger.info("Iniciando evaluación con contexto estático")
        static_results = evaluate(
            dataset=static_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
        )
        logger.info("Evaluación con contexto estático completada")
        logger.info(f"Resultados: {static_results}")
        
        logger.info("Iniciando evaluación con contexto dinámico")
        dynamic_results = evaluate(
            dataset=dynamic_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            run_config=my_run_config
        )
        logger.info("Evaluación con contexto dinámico completada")
        logger.info(f"Resultados: {dynamic_results}")
        
    except Exception as e:
        logger.error(f"Error en la evaluación: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())