# Informe de Evaluación RAG con RAGAS

## Resumen de Métricas

| Métrica | Valor |
|---------|-------|
| faithfulness | 0.5000 |
| response_relevancy | 0.5000 |
| context_precision | 0.5000 |
| context_recall | 0.5000 |
| harmfulness | 0.1000 |

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

- **Mejorar Faithfulness**: La fidelidad de las respuestas podría mejorarse. Considera ajustar el modelo para que se adhiera más estrictamente al contexto proporcionado.
- **Mejorar Response Relevancy**: Las respuestas podrían ser más relevantes a las preguntas. Considera refinar el proceso de generación de respuestas.
- **Mejorar Context Precision**: La precisión del contexto recuperado podría mejorarse. Considera ajustar el algoritmo de búsqueda vectorial o los embedding utilizados.
- **Mejorar Context Recall**: La cobertura del contexto podría mejorarse. Considera aumentar el número de chunks recuperados o mejorar la segmentación del contenido.
