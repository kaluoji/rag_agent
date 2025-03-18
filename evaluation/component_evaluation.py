import asyncio
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
from openai import AsyncOpenAI
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# Importar RAGAS para evaluación de componentes individuales
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall
)

# Importar tus agentes
from agents.ai_expert_v1 import ai_expert, AIDeps, get_embedding, debug_run_agent
from config.config import settings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class ComponentEvaluator:
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str):
        """
        Inicializa el evaluador de componentes RAG.
        
        Args:
            supabase_url: URL de Supabase
            supabase_key: Clave de Supabase
            openai_api_key: Clave API de OpenAI
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Crear dependencias para el agente AI Expert
        self.ai_deps = AIDeps(
            supabase=self.supabase,
            openai_client=self.openai_client
        )
        
        # Dataset para evaluación
        self.evaluation_dataset = None
    
    def load_dataset_from_json(self, json_path: str) -> None:
        """
        Carga el conjunto de datos de evaluación desde un archivo JSON.
        
        Args:
            json_path: Ruta al archivo JSON
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir el JSON a formato de evaluación
            questions = []
            answers = []
            contexts = []
            
            for key, item in data.items():
                questions.append(item.get('pregunta', ''))
                answers.append(item.get('respuesta', ''))
                contexts.append(item.get('contexto', ''))
            
            # Crear DataFrame
            self.evaluation_dataset = pd.DataFrame({
                'question': questions,
                'expected_answer': answers,
                'expected_context': contexts
            })
            
            logger.info(f"Dataset cargado con {len(questions)} ejemplos")
        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}")
            raise
    
    async def evaluate_retrieval_component(self, num_samples: int = None) -> pd.DataFrame:
        """
        Evalúa el componente de recuperación (retrieval) utilizando el agente real.
        
        Args:
            num_samples: Número de muestras a evaluar (None para todas)
        
        Returns:
            DataFrame con resultados de evaluación de retrieval
        """
        if self.evaluation_dataset is None:
            raise ValueError("Dataset no cargado. Llama a load_dataset_from_json primero.")
        
        # Tomar una muestra si se especifica
        if num_samples and num_samples < len(self.evaluation_dataset):
            eval_sample = self.evaluation_dataset.sample(num_samples, random_state=42)
        else:
            eval_sample = self.evaluation_dataset
        
        # Variables para almacenar resultados
        retrieved_contexts = []
        relevancy_scores = []
        recall_scores = []
        
        # Evaluar cada pregunta
        for idx, row in eval_sample.iterrows():
            question = row['question']
            expected_context = row['expected_context']
            
            logger.info(f"Evaluando retrieval para pregunta {idx}: {question[:50]}...")
            
            try:
                # Usar el mismo agente que usarías en producción
                response = await debug_run_agent(question, deps=self.ai_deps)
                
                # Extraer la respuesta generada
                generated_answer = response.data
                
                # Obtener el contexto usado mediante el mismo agente
                # Para esto, podríamos analizar los logs o usar una función auxiliar que simule
                # la misma recuperación que el agente
                
                # Por ahora usaremos el método anterior para obtener el contexto, pero con
                # la garantía de que pasamos por el mismo flujo que el agente
                context = await self._try_retrieve_with_query(question)
                
                # Si no encontramos contexto en el primer intento, intentar con una versión simplificada
                # Este enfoque simula lo que hace el agente internamente
                if not context:
                    # Intentar simplificar la consulta (similar a lo que hace el agente)
                    simplified_query = question.replace('¿', '').replace('?', '')
                    for word in ['cuál', 'cómo', 'qué', 'dónde', 'cuándo', 'quién', 'por qué', 'es', 'son', 'la', 'las', 'el', 'los', 'de', 'del']:
                        simplified_query = simplified_query.replace(f' {word} ', ' ')
                    
                    context = await self._try_retrieve_with_query(simplified_query)
                
                # Guardar resultados
                retrieved_contexts.append(context if context else "")
                
                # Calcular métricas
                relevancy = self._calculate_text_overlap(context if context else "", expected_context)
                relevancy_scores.append(relevancy)
                
                recall = self._calculate_recall(context if context else "", expected_context)
                recall_scores.append(recall)
                
            except Exception as e:
                logger.error(f"Error al evaluar retrieval para pregunta {idx}: {e}")
                retrieved_contexts.append("")
                relevancy_scores.append(0.0)
                recall_scores.append(0.0)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'question': eval_sample['question'].values,
            'retrieved_context': retrieved_contexts,
            'expected_context': eval_sample['expected_context'].values,
            'context_precision': relevancy_scores,
            'context_recall': recall_scores
        })
        
        # Calcular estadísticas
        avg_relevancy = np.mean(relevancy_scores)
        avg_recall = np.mean(recall_scores)
        
        logger.info(f"Evaluación de retrieval completada. Relevancia promedio: {avg_relevancy:.4f}, Recall promedio: {avg_recall:.4f}")
        
        return results_df

    async def _try_retrieve_with_query(self, query: str) -> str:
        """
        Intenta recuperar documentos con una consulta específica.
        
        Args:
            query: La consulta para buscar documentos
            
        Returns:
            El contexto recuperado, o cadena vacía si no se encontraron documentos
        """
        logger.info(f"Probando consulta: {query}")
        
        # Crear embedding para la consulta
        embedding = await get_embedding(query, self.openai_client)
        
        # Verificar embedding válido
        if not any(embedding):
            logger.warning(f"Embedding inválido para consulta: {query}")
            return ""
        
        # Búsqueda por similitud vectorial
        response = self.supabase.rpc(
            "match_visa_mastercard_v5", 
            {
                'query_embedding': embedding,
                'match_count': 8
            }
        ).execute()
        
        logger.info(f"Response data length: {len(response.data) if hasattr(response.data, '__len__') else 'N/A'}")
        
        # Si no hay resultados, retornar cadena vacía
        if not response.data or len(response.data) == 0:
            logger.info(f"No se encontraron documentos para consulta: {query}")
            return ""
        
        # Procesar resultados
        direct_matches = response.data
        logger.info(f"Encontrados {len(direct_matches)} chunks por similitud vectorial")
        
        # Búsqueda por clusters para enriquecer resultados
        cluster_ids = set()
        for doc in direct_matches:
            if 'metadata' in doc and doc['metadata'] and 'cluster_id' in doc['metadata']:
                cluster_id = doc['metadata'].get('cluster_id')
                if cluster_id is not None and cluster_id != -1:
                    cluster_ids.add(cluster_id)
        
        logger.info(f"Identificados {len(cluster_ids)} clusters diferentes: {cluster_ids}")
        
        # Buscar chunks adicionales por cluster
        cluster_responses = []
        for cluster_id in cluster_ids:
            try:
                cluster_result = self.supabase.rpc(
                    'match_visa_mastercard_v5_by_cluster',
                    {
                        'cluster_id': cluster_id,
                        'match_count': 4
                    }
                ).execute()
                
                if cluster_result.data:
                    cluster_docs_added = 0
                    for doc in cluster_result.data:
                        if not any(str(doc['id']) == str(existing_doc['id']) for existing_doc in direct_matches):
                            cluster_responses.append(doc)
                            cluster_docs_added += 1
                    logger.info(f"Recuperados {cluster_docs_added} chunks adicionales del cluster {cluster_id}")
            except Exception as e:
                logger.warning(f"Error al buscar en cluster {cluster_id}: {e}")
                continue
        
        # Formatear resultado
        all_chunks = direct_matches + cluster_responses
        
        # Eliminar duplicados
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                unique_chunks.append(chunk)
        
        # Construir contexto recuperado
        retrieved_context = ""
        for chunk in unique_chunks:
            retrieved_context += f"\n\n--- Fragmento ID: {chunk['id']} ---\n\n"
            retrieved_context += chunk["content"]
        
        logger.info(f"Devolviendo {len(direct_matches)} chunks directamente relevantes y {len(cluster_responses)} chunks relacionados por cluster.")
        
        return retrieved_context
    
    async def evaluate_generation_component(self, num_samples: int = None) -> pd.DataFrame:
        """
        Evalúa el componente de generación (generation) usando el agente real.
        
        Args:
            num_samples: Número de muestras a evaluar (None para todas)
        
        Returns:
            DataFrame con resultados de evaluación de generation
        """
        if self.evaluation_dataset is None:
            raise ValueError("Dataset no cargado. Llama a load_dataset_from_json primero.")
        
        # Tomar una muestra si se especifica
        if num_samples and num_samples < len(self.evaluation_dataset):
            eval_sample = self.evaluation_dataset.sample(num_samples, random_state=42)
        else:
            eval_sample = self.evaluation_dataset
        
        # Variables para almacenar resultados
        generated_answers = []
        faithfulness_scores = []
        relevancy_scores = []
        
        # Evaluar cada pregunta
        for idx, row in eval_sample.iterrows():
            question = row['question']
            expected_context = row['expected_context']
            expected_answer = row['expected_answer']
            
            logger.info(f"Evaluando generación para pregunta {idx}: {question[:50]}...")
            
            try:
                # Usar el agente real para generar respuestas
                response = await debug_run_agent(question, deps=self.ai_deps)
                
                # Extraer la respuesta generada
                generated_answer = response.data
                generated_answers.append(generated_answer)
                
                # Calcular métricas
                faithfulness_score = self._calculate_faithfulness(generated_answer, expected_context)
                faithfulness_scores.append(faithfulness_score)
                
                relevancy_score = self._calculate_response_relevancy(generated_answer, question, expected_answer)
                relevancy_scores.append(relevancy_score)
                
            except Exception as e:
                logger.error(f"Error al evaluar generación para pregunta {idx}: {e}")
                generated_answers.append("")
                faithfulness_scores.append(0.0)
                relevancy_scores.append(0.0)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'question': eval_sample['question'].values,
            'expected_context': eval_sample['expected_context'].values,
            'expected_answer': eval_sample['expected_answer'].values,
            'generated_answer': generated_answers,
            'faithfulness': faithfulness_scores,
            'response_relevancy': relevancy_scores
        })
        
        # Calcular estadísticas
        avg_faithfulness = np.mean(faithfulness_scores)
        avg_relevancy = np.mean(relevancy_scores)
        
        logger.info(f"Evaluación de generación completada. Fidelidad promedio: {avg_faithfulness:.4f}, Relevancia promedio: {avg_relevancy:.4f}")
        
        return results_df
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calcula una puntuación simplificada de solapamiento de texto.
        Esta es una métrica simplificada - RAGAS usa métodos más sofisticados.
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        
        Returns:
            Puntuación de solapamiento entre 0 y 1
        """
        # Dividir en palabras y convertir a conjuntos
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calcular intersección
        intersection = words1.intersection(words2)
        
        # Calcular solapamiento (similar a coeficiente de Jaccard)
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        return len(intersection) / len(words1.union(words2))
    
    def _calculate_recall(self, retrieved_text: str, expected_text: str) -> float:
        """
        Calcula una puntuación simplificada de recall.
        
        Args:
            retrieved_text: Texto recuperado
            expected_text: Texto esperado
        
        Returns:
            Puntuación de recall entre 0 y 1
        """
        # Dividir en palabras y convertir a conjuntos
        words_retrieved = set(retrieved_text.lower().split())
        words_expected = set(expected_text.lower().split())
        
        # Calcular intersección
        intersection = words_retrieved.intersection(words_expected)
        
        # Calcular recall
        if len(words_expected) == 0:
            return 0.0
        
        return len(intersection) / len(words_expected)
    
    def _calculate_faithfulness(self, generated_answer: str, context: str) -> float:
        """
        Calcula una puntuación simplificada de fidelidad.
        
        Args:
            generated_answer: Respuesta generada
            context: Contexto proporcionado
        
        Returns:
            Puntuación de fidelidad entre 0 y 1
        """
        # Dividir en palabras
        answer_words = set(generated_answer.lower().split())
        context_words = set(context.lower().split())
        
        # Calcular cuántas palabras de la respuesta están en el contexto
        overlap = sum(1 for word in answer_words if word in context_words)
        
        # Calcular proporción
        if len(answer_words) == 0:
            return 0.0
        
        return min(1.0, overlap / len(answer_words))
    
    def _calculate_response_relevancy(self, generated_answer: str, question: str, expected_answer: str) -> float:
        """
        Calcula una puntuación simplificada de relevancia de respuesta.
        
        Args:
            generated_answer: Respuesta generada
            question: Pregunta
            expected_answer: Respuesta esperada
        
        Returns:
            Puntuación de relevancia entre 0 y 1
        """
        # Combinamos la pregunta y la respuesta esperada
        reference = question + " " + expected_answer
        reference_words = set(reference.lower().split())
        
        # Palabras de la respuesta generada
        answer_words = set(generated_answer.lower().split())
        
        # Calcular solapamiento
        overlap = sum(1 for word in answer_words if word in reference_words)
        
        # Calcular proporción
        if len(answer_words) == 0:
            return 0.0
        
        return min(1.0, overlap / len(answer_words))
    
    def visualize_results(self, retrieval_results: pd.DataFrame, generation_results: pd.DataFrame, output_path: str = "rag_evaluation.png"):
        """
        Visualiza los resultados de la evaluación.
        
        Args:
            retrieval_results: Resultados de evaluación de retrieval
            generation_results: Resultados de evaluación de generation
            output_path: Ruta para guardar la visualización
        """
        plt.figure(figsize=(15, 10))
        
        # Configurar estilo
        sns.set(style="whitegrid")
        
        # Extraer métricas
        retrieval_metrics = {
            'Context Relevancy': retrieval_results['context_precision'].mean(),
            'Context Recall': retrieval_results['context_recall'].mean()
        }
        
        generation_metrics = {
            'Faithfulness': generation_results['faithfulness'].mean(),
            'Response Relevancy': generation_results['response_relevancy'].mean()
        }
        
        # Combinar métricas
        all_metrics = {**retrieval_metrics, **generation_metrics}
        metrics_df = pd.DataFrame({
            'metric': list(all_metrics.keys()),
            'value': list(all_metrics.values())
        })
        
        # Crear gráfico de barras (corregido)
        plt.subplot(2, 2, 1)
        sns.barplot(
            x='metric', 
            y='value', 
            data=metrics_df, 
            hue='metric',  # Añadir hue para evitar la advertencia
            palette="viridis", 
            legend=False  # No mostrar la leyenda ya que es redundante
        )
        plt.title('Métricas de Evaluación RAG')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Histograma de puntuaciones de relevancia de contexto
        plt.subplot(2, 2, 2)
        sns.histplot(retrieval_results['context_precision'], kde=True, bins=10)
        plt.title('Distribución de Relevancia de Contexto')
        plt.xlabel('Puntuación')
        plt.ylabel('Frecuencia')
        
        # Histograma de puntuaciones de recall de contexto
        plt.subplot(2, 2, 3)
        sns.histplot(retrieval_results['context_recall'], kde=True, bins=10)
        plt.title('Distribución de Recall de Contexto')
        plt.xlabel('Puntuación')
        plt.ylabel('Frecuencia')
        
        # Histograma de puntuaciones de fidelidad
        plt.subplot(2, 2, 4)
        sns.histplot(generation_results['faithfulness'], kde=True, bins=10)
        plt.title('Distribución de Fidelidad')
        plt.xlabel('Puntuación')
        plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Visualización guardada en {output_path}")
        
        # También generar un resumen en texto
        summary = f"""
        # Resumen de Evaluación de Componentes RAG
        
        ## Componente de Recuperación (Retrieval)
        - **Relevancia de Contexto**: {retrieval_metrics['Context Relevancy']:.4f}
        - **Recall de Contexto**: {retrieval_metrics['Context Recall']:.4f}
        
        ## Componente de Generación (Generation)
        - **Fidelidad**: {generation_metrics['Faithfulness']:.4f}
        - **Relevancia de Respuesta**: {generation_metrics['Response Relevancy']:.4f}
        
        ## Interpretación
        
        ### Retrieval
        - La relevancia del contexto mide cuán relacionado está el contexto recuperado con la pregunta.
        - El recall del contexto mide cuánta información relevante se ha recuperado.
        
        ### Generation
        - La fidelidad mide si la respuesta generada se mantiene fiel al contexto recuperado.
        - La relevancia de respuesta evalúa cuán bien la respuesta responde a la pregunta planteada.
        
        ## Recomendaciones
        
        - Si la relevancia o recall del contexto es baja, considere mejorar los embeddings o algoritmos de recuperación.
        - Si la fidelidad es baja, ajuste el modelo para que se adhiera mejor al contexto proporcionado.
        - Si la relevancia de respuesta es baja, mejore las instrucciones de generación.
        """
        
        with open("component_evaluation_summary.md", "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info("Resumen guardado en component_evaluation_summary.md")
    
    def generate_detailed_report(self, retrieval_results: pd.DataFrame, generation_results: pd.DataFrame, output_path: str = "detailed_evaluation_report.md"):
        """
        Genera un informe detallado de la evaluación de componentes.
        
        Args:
            retrieval_results: Resultados de evaluación de retrieval
            generation_results: Resultados de evaluación de generation
            output_path: Ruta para guardar el informe
        """
        # Calcular estadísticas
        retrieval_stats = {
            'context_precision_mean': retrieval_results['context_precision'].mean(),
            'context_precision_std': retrieval_results['context_precision'].std(),
            'context_recall_mean': retrieval_results['context_recall'].mean(),
            'context_recall_std': retrieval_results['context_recall'].std(),
        }
        
        generation_stats = {
            'faithfulness_mean': generation_results['faithfulness'].mean(),
            'faithfulness_std': generation_results['faithfulness'].std(),
            'response_relevancy_mean': generation_results['response_relevancy'].mean(),
            'response_relevancy_std': generation_results['response_relevancy'].std(),
        }
        
        # Identificar ejemplos de mejor y peor rendimiento
        best_retrieval_idx = retrieval_results['context_precision'].idxmax()
        worst_retrieval_idx = retrieval_results['context_precision'].idxmin()
        
        best_generation_idx = generation_results['faithfulness'].idxmax()
        worst_generation_idx = generation_results['faithfulness'].idxmin()
        
        # Generar informe
        report = f"""# Informe Detallado de Evaluación de Componentes RAG

## 1. Resumen Estadístico

### Componente de Recuperación (Retrieval)
- **Relevancia de Contexto**: 
  - Media: {retrieval_stats['context_precision_mean']:.4f}
  - Desviación estándar: {retrieval_stats['context_precision_std']:.4f}
- **Recall de Contexto**: 
  - Media: {retrieval_stats['context_recall_mean']:.4f}
  - Desviación estándar: {retrieval_stats['context_recall_std']:.4f}

### Componente de Generación (Generation)
- **Fidelidad**: 
  - Media: {generation_stats['faithfulness_mean']:.4f}
  - Desviación estándar: {generation_stats['faithfulness_std']:.4f}
- **Relevancia de Respuesta**: 
  - Media: {generation_stats['response_relevancy_mean']:.4f}
  - Desviación estándar: {generation_stats['response_relevancy_std']:.4f}

## 2. Análisis de Ejemplos

### Mejor Caso de Recuperación (Retrieval)
- **Pregunta**: {retrieval_results.loc[best_retrieval_idx, 'question']}
- **Relevancia**: {retrieval_results.loc[best_retrieval_idx, 'context_precision']:.4f}
- **Recall**: {retrieval_results.loc[best_retrieval_idx, 'context_recall']:.4f}

### Peor Caso de Recuperación (Retrieval)
- **Pregunta**: {retrieval_results.loc[worst_retrieval_idx, 'question']}
- **Relevancia**: {retrieval_results.loc[worst_retrieval_idx, 'context_precision']:.4f}
- **Recall**: {retrieval_results.loc[worst_retrieval_idx, 'context_recall']:.4f}

### Mejor Caso de Generación (Generation)
- **Pregunta**: {generation_results.loc[best_generation_idx, 'question']}
- **Fidelidad**: {generation_results.loc[best_generation_idx, 'faithfulness']:.4f}
- **Relevancia**: {generation_results.loc[best_generation_idx, 'response_relevancy']:.4f}

### Peor Caso de Generación (Generation)
- **Pregunta**: {generation_results.loc[worst_generation_idx, 'question']}
- **Fidelidad**: {generation_results.loc[worst_generation_idx, 'faithfulness']:.4f}
- **Relevancia**: {generation_results.loc[worst_generation_idx, 'response_relevancy']:.4f}

## 3. Distribución de Puntuaciones
- **Retrieval**: 
  - {len(retrieval_results[retrieval_results['context_precision'] > 0.8])} preguntas (${len(retrieval_results[retrieval_results['context_precision'] > 0.8]) / len(retrieval_results) * 100:.1f}$%) tienen relevancia de contexto > 0.8
  - {len(retrieval_results[retrieval_results['context_recall'] > 0.8])} preguntas (${len(retrieval_results[retrieval_results['context_recall'] > 0.8]) / len(retrieval_results) * 100:.1f}$%) tienen recall de contexto > 0.8

- **Generation**:
  - {len(generation_results[generation_results['faithfulness'] > 0.8])} respuestas (${len(generation_results[generation_results['faithfulness'] > 0.8]) / len(generation_results) * 100:.1f}$%) tienen fidelidad > 0.8
  - {len(generation_results[generation_results['response_relevancy'] > 0.8])} respuestas (${len(generation_results[generation_results['response_relevancy'] > 0.8]) / len(generation_results) * 100:.1f}$%) tienen relevancia > 0.8

## 4. Recomendaciones para Mejora

### Retrieval
- **Si la relevancia es baja**: 
  - Ajustar los embeddings utilizados para la búsqueda vectorial
  - Probar diferentes algoritmos de chunking para segmentar mejor el contenido
  - Aumentar la diversidad de los chunks recuperados
  - Considerar técnicas de expansión de consultas

- **Si el recall es bajo**:
  - Aumentar el número de chunks recuperados
  - Implementar técnicas de fusión de conjuntos de resultados
  - Considerar la recuperación híbrida (dense + sparse)

### Generation
- **Si la fidelidad es baja**:
  - Ajustar las instrucciones del modelo para enfatizar el uso exclusivo del contexto proporcionado
  - Experimentar con diferentes temperaturas (valores más bajos suelen aumentar la fidelidad)
  - Implementar post-procesamiento para verificar la presencia de información en el contexto

- **Si la relevancia es baja**:
  - Refinar las instrucciones de generación para enfocarse mejor en la pregunta
  - Considerar técnicas de reranking de contextos antes de la generación
  - Implementar un paso de verificación de relevancia antes de la respuesta final

"""
        
        # Guardar el informe
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Informe detallado guardado en {output_path}")

async def main():
    # Configuración desde variables de entorno o archivo de configuración
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializar evaluador
    evaluator = ComponentEvaluator(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        openai_api_key=openai_api_key
    )
    
    # Cargar dataset
    evaluator.load_dataset_from_json("evaluation/data/json.json")
    
    # Evaluar componentes individualmente
    # Puedes ajustar el número de muestras según necesites
    retrieval_results = await evaluator.evaluate_retrieval_component(num_samples=10)
    generation_results = await evaluator.evaluate_generation_component(num_samples=10)
    
    # Guardar resultados en CSV
    retrieval_results.to_csv("retrieval_evaluation.csv", index=False)
    generation_results.to_csv("generation_evaluation.csv", index=False)
    
    # Visualizar resultados
    evaluator.visualize_results(retrieval_results, generation_results)
    
    # Generar informe detallado
    evaluator.generate_detailed_report(retrieval_results, generation_results)
    
    logger.info("Evaluación de componentes completada")

if __name__ == "__main__":
    asyncio.run(main())