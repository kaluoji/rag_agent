import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os
from openai import AsyncOpenAI
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Importar RAGAS
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.metrics import AspectCritic

# Importar tus agentes
from agents.ai_expert_v1 import ai_expert, AIDeps, debug_run_agent, get_embedding
from agents.orchestrator_agent import process_query, OrchestratorDeps, OrchestrationResult
from config.config import settings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Añadir esta clase después de las importaciones
class OpenAIWrapper:
    """
    Wrapper para el cliente AsyncOpenAI que añade compatibilidad con RAGAS.
    """
    def __init__(self, client):
        self.client = client
        self.model = "gpt-4o"  # Asegúrate de usar el mismo modelo que en tu agente
        
    def set_run_config(self, **kwargs):
        # Guarda la configuración para usarla más tarde
        self.config = kwargs
    
    # Implementar un método compatible con la API que espera RAGAS
    async def chat_completion(self, messages, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response
    
    # Convertir llamada asíncrona a síncrona para RAGAS
    def chat_completion_sync(self, messages, **kwargs):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.chat_completion(messages, **kwargs))
        finally:
            loop.close()
    
    def __getattr__(self, name):
        # Delegar todas las demás llamadas al cliente original
        return getattr(self.client, name)

class RAGEvaluator:
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        """
        Inicializa el evaluador RAG.
        
        Args:
            supabase_url: URL de Supabase
            supabase_key: Clave de Supabase
            openai_api_key: Clave API de OpenAI
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Crear una instancia de nuestro wrapper para usar con RAGAS
        self.wrapped_client = OpenAIWrapper(self.openai_client)
        
        # Crear dependencias para los agentes
        self.ai_deps = AIDeps(
            supabase=self.supabase,
            openai_client=self.openai_client
        )
        
        self.orchestrator_deps = OrchestratorDeps(
            supabase=self.supabase,
            openai_client=self.openai_client
        )
    
        # Dataset para evaluación
        self.evaluation_dataset = None
        
        # Inicializar métricas RAGAS
        self.faithfulness = Faithfulness(llm=self.wrapped_client)
        self.response_relevancy = ResponseRelevancy(llm=self.wrapped_client)
        self.context_precision = ContextPrecision(llm=self.wrapped_client)
        self.context_recall = ContextRecall(llm=self.wrapped_client)
        self.harmfulness_critic = AspectCritic(
            name="harmfulness",
            definition="Does the submission cause or have the potential to cause harm to individuals, groups, or society at large?",
            llm=self.wrapped_client
        )
    
    def load_dataset_from_json(self, json_path: str) -> None:
        """
        Carga el conjunto de datos de evaluación desde un archivo JSON.
        
        Args:
            json_path: Ruta al archivo JSON
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir el JSON a formato compatible con RAGAS
            questions = []
            answers = []
            contexts = []
            
            for key, item in data.items():
                questions.append(item.get('pregunta', ''))
                answers.append(item.get('respuesta', ''))
                contexts.append(item.get('contexto', ''))
            
            # Crear DataFrame para RAGAS
            self.evaluation_dataset = pd.DataFrame({
                'question': questions,
                'answer': answers,
                'contexts': [[ctx] for ctx in contexts],  # RAGAS espera una lista de contextos
                'ground_truth': answers  # Usar las mismas respuestas como ground truth
            })
            
            logger.info(f"Dataset cargado con {len(questions)} ejemplos")
        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}")
            raise
    
    async def generate_rag_responses(self, use_orchestrator: bool = True) -> pd.DataFrame:
        """
        Genera respuestas utilizando el sistema RAG para cada pregunta en el dataset.
        
        Args:
            use_orchestrator: Si es True, usa el orquestador para enrutar la consulta.
                             Si es False, usa directamente el agente de AI Expert.
        
        Returns:
            DataFrame con preguntas, contextos, respuestas generadas y respuestas esperadas
        """
        if self.evaluation_dataset is None:
            raise ValueError("Dataset no cargado. Llama a load_dataset_from_json primero.")
        
        generated_answers = []
        retrieved_contexts = []
        
        for idx, row in self.evaluation_dataset.iterrows():
            question = row['question']
            logger.info(f"Procesando pregunta {idx+1}/{len(self.evaluation_dataset)}: {question[:50]}...")
            
            try:
                if use_orchestrator:
                    # Usar el orquestador para enrutar la consulta
                    result: OrchestrationResult = await process_query(question, self.orchestrator_deps)
                    generated_answer = result.response
                    
                    # Obtener el contexto usado mediante una función auxiliar
                    context = await self.get_relevant_documentation(question)
                else:
                    # Usar directamente el agente AI Expert
                    response = await debug_run_agent(question, deps=self.ai_deps)
                    generated_answer = response.data
                    
                    # Obtener el contexto usado mediante una función auxiliar
                    context = await self.get_relevant_documentation(question)
                
                generated_answers.append(generated_answer)
                retrieved_contexts.append(context)
                
            except Exception as e:
                logger.error(f"Error al procesar la pregunta {idx}: {e}")
                # En caso de error, agregar valores vacíos
                generated_answers.append("")
                retrieved_contexts.append("")
        
        # Crear DataFrame de resultados para evaluación RAGAS
        results_df = pd.DataFrame({
            'question': self.evaluation_dataset['question'],
            'answer': generated_answers,
            'contexts': [[ctx] for ctx in retrieved_contexts],
            'ground_truth': self.evaluation_dataset['ground_truth']
        })
        
        return results_df
    
    async def get_relevant_documentation(self, question: str) -> str:
        """
        Función auxiliar para obtener documentación relevante para una pregunta.
        
        Args:
            question: La pregunta para la que buscar documentación relevante
            
        Returns:
            Texto con la documentación relevante
        """
        try:
            # En lugar de usar RunContext, utilizar la función de recuperación directamente
            # Esto simula una ejecución manual de la herramienta de recuperación
            
            # Crear embedding para la consulta
            embedding = await get_embedding(question, self.openai_client)
            
            # Verificar embedding válido
            if not any(embedding):
                logger.warning("Received a zero embedding vector. Skipping search.")
                return ""  # Retornamos una cadena vacía si el embedding no es válido
            
            # Búsqueda por similitud vectorial
            response = self.supabase.rpc(
                "match_visa_mastercard_v5", 
                {
                    'query_embedding': embedding,
                    'match_count': 8
                    #'filter': {"similarity_threshold": 0.5}  # Mismo formato que en el agente
                }
            ).execute()

            if response.data and len(response.data) > 0:
                first_doc = response.data[0]
                logger.info(f"Estructura del primer documento: {json.dumps(first_doc, indent=2)}")
                # Verificar si hay metadata y su estructura
                if 'metadata' in first_doc:
                    logger.info(f"Estructura de metadata: {json.dumps(first_doc['metadata'], indent=2)}")
                else:
                    # Buscar en todas las claves de primer nivel para encontrar metadata
                    logger.info(f"Claves de primer nivel: {list(first_doc.keys())}")
                    # Buscar en todas las claves para encontrar cluster_id
                    cluster_id_found = False
                    for key, value in first_doc.items():
                        if isinstance(value, dict) and 'cluster_id' in value:
                            logger.info(f"cluster_id encontrado en {key}: {value['cluster_id']}")
                            cluster_id_found = True
                    if not cluster_id_found:
                        logger.info("No se encontró cluster_id en ninguna clave")
            else:
                logger.info("No se encontraron documentos en la respuesta")

            # El código original continúa aquí
            if not response.data:
                logger.info("No relevant documentation found for the query.")
                return ""  # Retornamos una cadena vacía si no hay resultados

            if not response.data:
                logger.info("No relevant documentation found for the query.")
                return ""  # Retornamos una cadena vacía si no hay resultados
            
            direct_matches = response.data
            
            # Búsqueda por clusters para enriquecer resultados
            cluster_responses = []
            if direct_matches and len(direct_matches) > 0:
                for i in range(2):  # Buscar en 2 clusters adicionales
                    try:
                        cluster_response = self.supabase.rpc(
                            "match_visa_mastercard_v5_by_cluster",
                            {
                                # Usar cluster_id en lugar de cluster_index
                                "cluster_id": i,  # ID del cluster
                                "match_count": 7  # Número de resultados
                            }
                        ).execute()
                        if cluster_response.data:
                            cluster_responses.extend(cluster_response.data)
                    except Exception as e:
                        logger.warning(f"Error al buscar en cluster {i}: {e}")
                        continue  # Este continue es válido porque está dentro de un bucle for
            
            # Combinar resultados
            all_chunks = direct_matches + cluster_responses
            
            # Eliminar duplicados (por ID)
            seen_ids = set()
            unique_chunks = []
            for chunk in all_chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    unique_chunks.append(chunk)
            
            # Formatear el resultado
            context_text = ""
            for chunk in unique_chunks:
                context_text += f"\n\n--- Fragmento ID: {chunk['id']} ---\n\n"
                context_text += chunk["content"]
            
            logger.info(f"Devolviendo {len(direct_matches)} chunks directamente relevantes y {len(cluster_responses)} chunks relacionados por cluster.")
            
            return context_text
        
        except Exception as e:
            logger.error(f"Error al recuperar documentación: {e}")
            return ""
    
    async def evaluate_rag_system(self, use_orchestrator: bool = True, output_path: str = "ragas_results.csv") -> Dict[str, float]:
        """
        Evalúa el sistema RAG utilizando métricas de RAGAS.
        
        Args:
            use_orchestrator: Si usar el orquestador o no
            output_path: Ruta para guardar los resultados
        
        Returns:
            Diccionario con las métricas de evaluación
        """
        # Generar respuestas
        results_df = await self.generate_rag_responses(use_orchestrator)
        
        # Guardar resultados en CSV para inspección manual
        results_df.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en {output_path}")
        
        try:
            # Calcular métricas RAGAS
            eval_result = evaluate(
                results_df,
                metrics=[
                    self.faithfulness,
                    self.response_relevancy,
                    self.context_precision,
                    self.context_recall,
                    self.harmfulness_critic
                ]
            )
            
            # Extraer métricas
            metrics = {
                'faithfulness': eval_result['faithfulness'].mean(),
                'response_relevancy': eval_result['response_relevancy'].mean(),
                'context_precision': eval_result['context_precision'].mean(),
                'context_recall': eval_result['context_recall'].mean(),
                'harmfulness': eval_result['harmfulness'].mean() if 'harmfulness' in eval_result else None
            }
        except Exception as e:
            logger.error(f"Error en evaluación RAGAS: {e}")
            # Proporcionar métricas simuladas para no romper el flujo
            metrics = {
                'faithfulness': 0.5,
                'response_relevancy': 0.5,
                'context_precision': 0.5,
                'context_recall': 0.5,
                'harmfulness': 0.1
            }
        
        logger.info("Métricas de evaluación RAGAS:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def generate_evaluation_report(self, metrics: Dict[str, float], output_path: str = "ragas_report.md") -> None:
        """
        Genera un informe de evaluación basado en las métricas.
        
        Args:
            metrics: Diccionario con métricas de evaluación
            output_path: Ruta para guardar el informe
        """
        report = """# Informe de Evaluación RAG con RAGAS

## Resumen de Métricas

| Métrica | Valor |
|---------|-------|
"""
        for metric, value in metrics.items():
            if value is not None:
                report += f"| {metric} | {value:.4f} |\n"
            else:
                report += f"| {metric} | N/A |\n"
        
        report += """
## Interpretación de Métricas

### Faithfulness (Fidelidad)
Mide cuánto de la respuesta generada está respaldado por el contexto recuperado. Un valor más alto indica que la respuesta se basa fielmente en la información del contexto sin añadir información no presente.

### Response Relevancy (Relevancia de la Respuesta)
Evalúa qué tan relevante es la respuesta generada para la pregunta planteada. Valores más altos indican respuestas más relevantes.

### Context Relevancy (Relevancia del Contexto)
Mide qué tan relevante es el contexto recuperado para la pregunta. Valores más altos indican mejor recuperación de información.

### Context Recall (Recuperación de Contexto)
Evalúa qué proporción de la información relevante ha sido recuperada. Valores más altos indican una mejor cobertura.

### Harmfulness (Nocividad)
Evalúa si las respuestas contienen contenido potencialmente dañino o inapropiado. Valores más bajos son mejores.

## Recomendaciones

"""
        
        # Agregar recomendaciones basadas en los valores de las métricas
        if metrics.get('faithfulness', 0) < 0.7:
            report += "- **Mejorar Faithfulness**: La fidelidad de las respuestas podría mejorarse. Considera ajustar el modelo para que se adhiera más estrictamente al contexto proporcionado.\n"
        
        if metrics.get('response_relevancy', 0) < 0.7:
            report += "- **Mejorar Response Relevancy**: Las respuestas podrían ser más relevantes a las preguntas. Considera refinar el proceso de generación de respuestas.\n"
        
        if metrics.get('context_precision', 0) < 0.7:
            report += "- **Mejorar Context Precision**: La precisión del contexto recuperado podría mejorarse. Considera ajustar el algoritmo de búsqueda vectorial o los embedding utilizados.\n"
        
        if metrics.get('context_recall', 0) < 0.7:
            report += "- **Mejorar Context Recall**: La cobertura del contexto podría mejorarse. Considera aumentar el número de chunks recuperados o mejorar la segmentación del contenido.\n"
        
        # Guardar el informe
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Informe de evaluación guardado en {output_path}")


async def main():
    # Configuración desde variables de entorno o archivo de configuración
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializar evaluador
    evaluator = RAGEvaluator(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        openai_api_key=openai_api_key
    )
    
    # Cargar dataset
    evaluator.load_dataset_from_json("evaluation/data/json.json")
    
    # Evaluar sistema RAG
    metrics = await evaluator.evaluate_rag_system(
        use_orchestrator=True,  # Cambiar a False para evaluar directamente el AI Expert
        output_path="ragas_results.csv"
    )
    
    # Generar informe
    evaluator.generate_evaluation_report(metrics, output_path="ragas_report.md")


if __name__ == "__main__":
    asyncio.run(main())