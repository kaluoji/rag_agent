# Informe Detallado de Evaluación de Componentes RAG

## 1. Resumen Estadístico

### Componente de Recuperación (Retrieval)
- **Relevancia de Contexto**: 
  - Media: 0.0000
  - Desviación estándar: 0.0000
- **Recall de Contexto**: 
  - Media: 0.0000
  - Desviación estándar: 0.0000

### Componente de Generación (Generation)
- **Fidelidad**: 
  - Media: 0.5224
  - Desviación estándar: 0.1044
- **Relevancia de Respuesta**: 
  - Media: 0.4120
  - Desviación estándar: 0.0350

## 2. Análisis de Ejemplos

### Mejor Caso de Recuperación (Retrieval)
- **Pregunta**: ¿Qué impacto tiene el aumento de las tarifas de aceptación de Visa en comercios de alto riesgo?
- **Relevancia**: 0.0000
- **Recall**: 0.0000

### Peor Caso de Recuperación (Retrieval)
- **Pregunta**: ¿Qué impacto tiene el aumento de las tarifas de aceptación de Visa en comercios de alto riesgo?
- **Relevancia**: 0.0000
- **Recall**: 0.0000

### Mejor Caso de Generación (Generation)
- **Pregunta**: ¿Qué impacto tiene la nueva tasa de acceso a la Sala Visa Infinite Lounge en Brasil y cuál es el incremento porcentual?
- **Fidelidad**: 0.6792
- **Relevancia**: 0.4340

### Peor Caso de Generación (Generation)
- **Pregunta**: ¿Cuáles son las nuevas tasas de intercambio (IRF) aplicables en 2025?
- **Fidelidad**: 0.3590
- **Relevancia**: 0.3846

## 3. Distribución de Puntuaciones
- **Retrieval**: 
  - 0 preguntas ($0.0$%) tienen relevancia de contexto > 0.8
  - 0 preguntas ($0.0$%) tienen recall de contexto > 0.8

- **Generation**:
  - 0 respuestas ($0.0$%) tienen fidelidad > 0.8
  - 0 respuestas ($0.0$%) tienen relevancia > 0.8

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

