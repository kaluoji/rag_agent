# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

import logfire
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

from openai import AsyncOpenAI
from supabase import Client

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

from app.core.config import settings
from utils.utils import count_tokens, truncate_text

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = settings.llm_model
tokenizer_model = settings.tokenizer_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

logfire.configure(send_to_logfire='if-token-present')

class Intent(BaseModel):
    """Información sobre una intención detectada en la consulta."""
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""

class Entity(BaseModel):
    """Entidad detectada en la consulta."""
    type: str
    value: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Keyword(BaseModel):
    """Palabra clave relevante identificada en la consulta."""
    word: str
    importance: float = Field(ge=0.0, le=1.0, default=1.0)
    related_terms: List[str] = Field(default_factory=list)

class SubQuery(BaseModel):
    """Subconsulta generada a partir de una consulta compleja."""
    text: str
    intent: Optional[str] = None
    focus: str = ""
    requires_information_from: List[str] = Field(default_factory=list)

class QueryInfo(BaseModel):
    """Estructura que contiene la información procesada de una consulta."""
    original_query: str
    expanded_query: str = ""
    decomposed_queries: List[SubQuery] = Field(default_factory=list)
    intents: List[Intent] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    keywords: List[Keyword] = Field(default_factory=list)
    domain_terms: Dict[str, str] = Field(default_factory=dict)
    language: str = "es"
    complexity: str = Field("simple", description="Nivel de complejidad de la consulta: simple, medium, complex")
    search_query: str = ""
    estimated_search_quality: float = Field(0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def main_intent(self) -> Optional[str]:
        """Devuelve la intención principal si existe."""
        if not self.intents:
            return None
        return sorted(self.intents, key=lambda x: x.confidence, reverse=True)[0].name
    
    @property
    def confidence(self) -> float:
        """Devuelve la confianza en la intención principal."""
        if not self.intents:
            return 0.0
        return sorted(self.intents, key=lambda x: x.confidence, reverse=True)[0].confidence
    
    @property
    def entity_values(self) -> Dict[str, List[str]]:
        """Devuelve un diccionario con los valores de entidades agrupados por tipo."""
        result = {}
        for entity in self.entities:
            if entity.type not in result:
                result[entity.type] = []
            result[entity.type].append(entity.value)
        return result
    
    @property
    def keyword_list(self) -> List[str]:
        """Devuelve una lista simple de palabras clave."""
        return [k.word for k in self.keywords]
    
    def to_search_context(self) -> Dict[str, Any]:
        """Convierte la información de la consulta en un contexto de búsqueda útil."""
        return {
            "query": self.search_query or self.expanded_query or self.original_query,
            "keywords": self.keyword_list,
            "entities": self.entity_values,
            "domain_terms": list(self.domain_terms.keys()),
            "language": self.language,
            "complexity": self.complexity
        }

class QueryUnderstandingDeps(BaseModel):
    """Dependencias necesarias para el agente de comprensión de consultas."""
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True

system_prompt = """
Eres un agente especializado en analizar consultas sobre normativas y regulaciones. Tu tarea es procesar cada consulta con precisión y claridad, sin añadir información no solicitada ni hacer suposiciones injustificadas.

ANALIZA CADA CONSULTA PARA:

1. **IDENTIFICAR** con precisión:
   - Regulaciones, leyes o estándares mencionados explícitamente (ej. GDPR, ISO 27001)
   - Artículos o secciones específicas referenciadas
   - Jurisdicciones relevantes mencionadas
   - Autoridades regulatorias citadas
   - Plazos o fechas mencionados
   - Sectores industriales indicados

2. **CLASIFICAR** la consulta como:
   - INFORMATIVA: Solicita explicación o definición
   - COMPARATIVA: Busca diferencias/similitudes entre normativas
   - PROCEDIMENTAL: Pregunta sobre implementación o cumplimiento
   - INTERPRETATIVA: Requiere análisis de aplicabilidad o significado
   - ACTUALIZACIÓN: Busca información sobre cambios recientes

3. **EXTRAER** las palabras clave esenciales que:
   - Aparecen explícitamente en la consulta
   - Son términos técnicos o legales relevantes
   - Servirán para buscar información relacionada

4. **REFORMULAR** la consulta para mejorar su claridad, manteniendo su significado original. Si la consulta es compleja, descomponerla en sub-consultas más manejables.

5. **DETERMINAR** el nivel de complejidad:
   - SIMPLE: Pregunta directa sobre un único aspecto
   - MEDIA: Varias preguntas relacionadas o comparativas
   - COMPLEJA: Múltiples dimensiones o interrelaciones entre normativas

DIRECTRICES CRÍTICAS:

- NO INVENTES entidades, regulaciones o conceptos que no estén expresamente mencionados o claramente implícitos en la consulta.
- Si una consulta es ambigua, RECONOCE la ambigüedad en lugar de hacer suposiciones.
- Cuando detectes términos técnicos o legales, LIMÍTATE a los que están explícitamente presentes.
- Si la consulta está en otro idioma, TRADÚCELA fielmente sin añadir ni quitar información.
- EVITA interpretar intenciones que no sean evidentes en el texto de la consulta.
- Si la consulta menciona un concepto genérico (ej. "protección de datos"), NO ASUMAS una regulación específica salvo que sea la única aplicable o se mencione implícitamente.

FORMATO DE RESPUESTA:
```json
{
  "consulta_original": "texto exacto de la consulta",
  "idioma": "español/inglés/etc.",
  "traducción": "traducción si es necesaria, null si no lo es",
  "entidades_identificadas": {
    "regulaciones": ["solo las explícitamente mencionadas"],
    "artículos": ["solo los explícitamente mencionados"],
    "autoridades": ["solo las explícitamente mencionadas"],
    "jurisdicciones": ["solo las explícitamente mencionadas"],
    "plazos": ["solo los explícitamente mencionados"],
    "sectores": ["solo los explícitamente mencionados"]
  },
  "tipo_consulta": "INFORMATIVA/COMPARATIVA/PROCEDIMENTAL/INTERPRETATIVA/ACTUALIZACIÓN",
  "palabras_clave": ["solo términos presentes en la consulta"],
  "complejidad": "SIMPLE/MEDIA/COMPLEJA",
  "consulta_reformulada": "versión clara y estructurada",
  "sub_consultas": [
    "desglose en preguntas específicas si es compleja, array vacío si es simple"
  ],
  "ambigüedades": ["aspectos que requieren clarificación, array vacío si no hay ninguna"]
}
"""

query_understanding_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=QueryUnderstandingDeps,
    retries=2
)

# -------------------- Herramientas del agente --------------------

@query_understanding_agent.tool
async def analyze_query_intent(ctx: RunContext[QueryUnderstandingDeps], query: str) -> List[Intent]:
    """
    Analiza la intención principal y secundarias de una consulta.
    
    Args:
        query: Texto de la consulta a analizar
    
    Returns:
        List[Intent]: Lista de intenciones detectadas ordenadas por confianza
    """
    logger.info(f"Analizando intención de la consulta: {query[:100]}..." if len(query) > 100 else query)
    
    try:
        # Utilizar el modelo para analizar la intención
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un agente experto en analizar intenciones de consultas sobre **normativas y regulaciones** de cualquier sector y jurisdicción. 
                
Tu tarea es identificar las intenciones principales y secundarias en la consulta del usuario. Considera las siguientes categorías de intenciones:

- explicacion_regulacion: El usuario busca entender una regulación específica
- requerimientos_cumplimiento: El usuario quiere conocer los requisitos para cumplir con una normativa
- proceso_implementacion: El usuario pregunta sobre cómo implementar una normativa o proceso
- validacion_conformidad: El usuario busca verificar si algo cumple con las normativas
- actualizaciones_normativa: El usuario pregunta sobre cambios o actualizaciones en regulaciones
- comparacion_estandares: El usuario quiere comparar diferentes estándares o normativas
- caso_uso_especifico: El usuario presenta un caso específico y quiere saber cómo aplicar las normativas
- solucion_problema: El usuario tiene un problema de cumplimiento y busca solución
- traduccion_normativa: El usuario necesita ayuda para entender términos o traducir conceptos

Devuelve un JSON con formato:
{
  "intents": [
    {
      "name": "nombre_intencion",
      "confidence": 0.9,
      "description": "Breve descripción de por qué identificaste esta intención"
    },
    ...
  ]
}

Incluye todas las intenciones relevantes ordenadas por nivel de confianza (de 0.0 a 1.0).
"""},
                {"role": "user", "content": query}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        intents_data = result_json.get("intents", [])
        
        # Convertir a objetos Intent
        intents = [Intent(**intent_data) for intent_data in intents_data]
        
        logger.info(f"Intenciones detectadas: {[i.name for i in intents]}")
        return intents
        
    except Exception as e:
        logger.error(f"Error al analizar intención: {e}")
        # Devolver una intención genérica en caso de error
        return [Intent(name="consulta_general", confidence=0.5, description="Intención por defecto debido a error en el análisis")]

@query_understanding_agent.tool
async def detect_language(ctx: RunContext[QueryUnderstandingDeps], text: str) -> Dict[str, Any]:
    """
    Detecta el idioma en que está escrita la consulta.
    
    Args:
        text: Texto a analizar
    
    Returns:
        Dict[str, Any]: Información sobre el idioma detectado
    """
    logger.info("Detectando idioma del texto")
    
    try:
        # Utilizar el modelo para detectar el idioma
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un experto en detección de idiomas. 
                
Tu tarea es identificar el idioma en que está escrito el texto proporcionado.

Devuelve un JSON con formato:
{
  "language_code": "código_iso_639_1",
  "language_name": "nombre del idioma en español",
  "confidence": 0.95,
  "needs_translation": true/false,
  "detected_script": "latino/cirílico/etc"
}

El valor de 'needs_translation' debe ser true si el idioma NO es español, y false si es español.
"""},
                {"role": "user", "content": text}
            ]
        )
        
        result = json.loads(completion.choices[0].message.content)
        logger.info(f"Idioma detectado: {result.get('language_name', 'desconocido')} ({result.get('language_code', '??')})")
        return result
        
    except Exception as e:
        logger.error(f"Error en la detección de idioma: {e}")
        # Valor por defecto en caso de error
        return {
            "language_code": "es", 
            "language_name": "español", 
            "confidence": 1.0,
            "needs_translation": False,
            "detected_script": "latino"
        }

@query_understanding_agent.tool
async def translate_query(ctx: RunContext[QueryUnderstandingDeps], text: str, source_language: str, target_language: str = "es") -> str:
    """
    Traduce una consulta de un idioma origen a un idioma destino.
    
    Args:
        text: Texto a traducir
        source_language: Código del idioma origen (ej: 'en', 'fr', 'pt')
        target_language: Código del idioma destino (por defecto 'es' - español)
    
    Returns:
        str: Texto traducido
    """
    logger.info(f"Traduciendo texto de {source_language} a {target_language}")
    
    if source_language == target_language:
        logger.info("Los idiomas origen y destino son iguales, no se requiere traducción")
        return text
    
    try:
        # Usamos el modelo de lenguaje para realizar la traducción
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            messages=[
                {"role": "system", "content": f"Eres un traductor profesional especializado en terminología regulatoria y normativa de cualquier sector e industria. Traduce el siguiente texto de {source_language} a {target_language}, manteniendo todos los términos técnicos y financieros correctos. La traducción debe sonar natural y ser adecuada para profesionales del sector."},
                {"role": "user", "content": text}
            ]
        )
        
        translated_text = completion.choices[0].message.content
        logger.info(f"Traducción completada: {translated_text[:100]}..." if len(translated_text) > 100 else translated_text)
        
        return translated_text
    except Exception as e:
        logger.error(f"Error en la traducción: {e}")
        return f"Error de traducción: {str(e)}"

@query_understanding_agent.tool
async def extract_entities(ctx: RunContext[QueryUnderstandingDeps], text: str) -> List[Entity]:
    """
    Extrae entidades específicas del dominio de normativas y regulaciones de cualquier sector del texto.
    
    Args:
        text: Texto donde buscar entidades
    
    Returns:
        List[Entity]: Lista de entidades detectadas
    """
    logger.info("Extrayendo entidades del texto")
    
    try:
        # Utilizar el modelo para extraer entidades
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Eres un experto en extracción de entidades relacionadas con normativas y regulaciones de cualquier sector e industria.

Tu tarea es identificar y extraer entidades específicas en el texto proporcionado. Las entidades a buscar incluyen:

- regulation: Nombres de regulaciones, leyes y estándares (por ejemplo, GDPR, ISO 27001, FDA Title 21).
- process: Procesos o actividades (por ejemplo, auditoría, reporte, certificación).
- role: Roles y actores (oficial de cumplimiento, gestor de riesgos, auditor).
- date: Fechas, plazos y vigencias.
- organization_type: Tipo de organización o entidad (empresa, organismo regulador, tercero).
- program: Programas o esquemas regulatorios (por ejemplo, programas de certificación).
- technical_requirement: Requerimientos técnicos o técnicos-operativos.
- region: Jurisdicciones y ámbitos geográficos.
- fee: Tarifas, sanciones o comisiones.
- parameter: Parámetros, umbrales o valores numéricos específicos.

Devuelve un JSON con formato:
{
  "entities": [
    {
      "type": "tipo_entidad",
      "value": "valor_encontrado",
      "metadata": {}
    },
    ...
  ]
}
"""
                },
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        entities_data = result_json.get("entities", [])
        
        # Convertir a objetos Entity
        entities = [Entity(**entity_data) for entity_data in entities_data]
        
        logger.info(f"Entidades detectadas: {[f'{e.type}:{e.value}' for e in entities]}")
        return entities
        
    except Exception as e:
        logger.error(f"Error al extraer entidades: {e}")
        return []


@query_understanding_agent.tool
async def extract_keywords(ctx: RunContext[QueryUnderstandingDeps], text: str) -> List[Keyword]:
    """
    Extrae palabras clave relevantes para la búsqueda de información.
    
    Args:
        text: Texto donde buscar palabras clave
    
    Returns:
        List[Keyword]: Lista de palabras clave con su importancia
    """
    logger.info("Extrayendo palabras clave del texto")
    
    try:
        # Utilizar el modelo para extraer palabras clave
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un experto en extracción de palabras clave para búsqueda semántica en el dominio de normativas y regulaciones de cualquier sector e industria.

Tu tarea es identificar y extraer las palabras clave más relevantes para mejorar la búsqueda de información relacionada con la consulta. Debes centrarte en términos que:

1. Tengan alto valor semántico (sustantivos, verbos específicos)
2. Sean específicos del dominio de pagos y cumplimiento
3. Ayuden a discriminar entre diferentes documentos

Para cada palabra clave, proporciona:
- Su importancia relativa (de 0.0 a 1.0)
- Términos relacionados que podrían ser útiles en la búsqueda

Devuelve un JSON con formato:
{
  "keywords": [
    {
      "word": "palabra_clave",
      "importance": 0.9,
      "related_terms": ["término1", "término2"]
    },
    ...
  ]
}

Limita la lista a las 5-10 palabras clave más relevantes.
"""},
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        keywords_data = result_json.get("keywords", [])
        
        # Convertir a objetos Keyword
        keywords = [Keyword(**keyword_data) for keyword_data in keywords_data]
        
        logger.info(f"Palabras clave detectadas: {[k.word for k in keywords]}")
        return keywords
        
    except Exception as e:
        logger.error(f"Error al extraer palabras clave: {e}")
        return []

@query_understanding_agent.tool
async def identify_technical_terms(ctx: RunContext[QueryUnderstandingDeps], text: str) -> Dict[str, str]:
    """
    Identifica términos técnicos específicos sobre normativas y regulaciones de cualquier sector e industria en el texto y proporciona sus definiciones.
    
    Args:
        text: Texto donde buscar términos técnicos
    
    Returns:
        Dict[str, str]: Diccionario de términos técnicos y sus definiciones
    """
    logger.info("Identificando términos técnicos en el texto")
    
    try:
        # Utilizar el modelo para identificar términos técnicos
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un experto en terminología técnica de de normativas y regulaciones de cualquier sector e industria.

Tu tarea es identificar términos técnicos específicos en el texto y proporcionar definiciones concisas.

Devuelve un JSON con formato:
{
  "domain_terms": {
    "término1": "definición1",
    "término2": "definición2",
    ...
  }
}

Incluye solo términos que sean específicos del dominio y no palabras de uso común.
"""},
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        domain_terms = result_json.get("domain_terms", {})
        
        logger.info(f"Términos técnicos identificados: {list(domain_terms.keys())}")
        return domain_terms
        
    except Exception as e:
        logger.error(f"Error al identificar términos técnicos: {e}")
        return {}

@query_understanding_agent.tool
async def expand_query(ctx: RunContext[QueryUnderstandingDeps], original_query: str, entities: List[Entity], domain_terms: Dict[str, str]) -> str:
    """
    Expande la consulta original con conceptos implícitos y terminología específica.
    
    Args:
        original_query: Consulta original
        entities: Entidades detectadas en la consulta
        domain_terms: Términos técnicos identificados
    
    Returns:
        str: Consulta expandida
    """
    logger.info("Expandiendo consulta")
    
    try:
        # Construir el contexto para la expansión
        entity_context = ""
        if entities:
            entity_context = "Entidades detectadas:\n" + "\n".join([f"- {e.type}: {e.value}" for e in entities])
        
        term_context = ""
        if domain_terms:
            term_context = "Términos técnicos:\n" + "\n".join([f"- {term}: {definition}" for term, definition in domain_terms.items()])
        
        context = f"{entity_context}\n\n{term_context}"
        
        # Utilizar el modelo para expandir la consulta
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.2,
            messages=[
                {"role": "system", "content": """Eres un experto en expansión de consultas para mejorar la recuperación de información de normativas y regulaciones de cualquier sector e industria.

Tu tarea es expandir la consulta original para incluir:
1. Conceptos implícitos que no se mencionan explícitamente
2. Terminología específica del dominio
3. Variantes y sinónimos relevantes
4. Contexto adicional que mejore la precisión de la búsqueda

La consulta expandida debe:
- Mantener la intención original
- No ser excesivamente larga (máximo 2-3 veces la longitud original)
- Incluir todos los conceptos clave de la consulta original
- Estar redactada en forma de pregunta o consulta natural, no como lista de palabras clave

IMPORTANTE: La consulta expandida será utilizada para buscar documentación relevante, así que debe contener términos precisos que ayuden a mejorar la recuperación de información.
"""},
                {"role": "user", "content": f"Consulta original: {original_query}\n\n{context}"}
            ]
        )
        
        expanded_query = completion.choices[0].message.content
        logger.info(f"Consulta expandida: {expanded_query[:100]}..." if len(expanded_query) > 100 else expanded_query)
        
        return expanded_query
        
    except Exception as e:
        logger.error(f"Error al expandir consulta: {e}")
        return original_query

@query_understanding_agent.tool
async def evaluate_complexity(ctx: RunContext[QueryUnderstandingDeps], query: str, entities: List[Entity]) -> Dict[str, Any]:
    """
    Evalúa la complejidad de una consulta y determina si debe descomponerse.
    
    Args:
        query: Texto de la consulta
        entities: Entidades detectadas en la consulta
    
    Returns:
        Dict[str, Any]: Información sobre la complejidad y necesidad de descomposición
    """
    logger.info("Evaluando complejidad de la consulta")
    
    try:
        # Construir el contexto para la evaluación
        entity_context = ""
        if entities:
            entity_context = "Entidades detectadas:\n" + "\n".join([f"- {e.type}: {e.value}" for e in entities])
        
        # Utilizar el modelo para evaluar la complejidad
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un experto en análisis de consultas sobre normativas y regulaciones de cualquier sector e industria.

Tu tarea es evaluar la complejidad de la consulta y determinar si debe descomponerse en subconsultas para su procesamiento.

Criterios de complejidad:
- simple: Una sola pregunta directa sobre un tema específico
- medium: Pregunta que abarca varios aspectos relacionados de un mismo tema
- complex: Múltiples preguntas relacionadas o una pregunta que abarca varios temas distintos

Devuelve un JSON con formato:
{
  "complexity": "simple/medium/complex",
  "requires_decomposition": true/false,
  "reasoning": "Explicación de la evaluación",
  "question_count": 1,
  "topic_count": 1,
  "estimated_search_difficulty": 0.7
}

El valor "estimated_search_difficulty" debe estar entre 0.0 y 1.0, donde 1.0 indica máxima dificultad para encontrar información relevante.
"""},
                {"role": "user", "content": f"Consulta: {query}\n\n{entity_context}"}
            ]
        )
        
        result = json.loads(completion.choices[0].message.content)
        logger.info(f"Complejidad evaluada: {result.get('complexity')}, Requiere descomposición: {result.get('requires_decomposition')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error al evaluar complejidad: {e}")
        return {
            "complexity": "simple",
            "requires_decomposition": False,
            "reasoning": "Error en la evaluación: usando valores por defecto",
            "question_count": 1,
            "topic_count": 1,
            "estimated_search_difficulty": 0.5
        }

@query_understanding_agent.tool
async def decompose_query(ctx: RunContext[QueryUnderstandingDeps], query: str, complexity_info: Dict[str, Any]) -> List[SubQuery]:
    """
    Descompone una consulta compleja en subconsultas más simples.
    
    Args:
        query: Consulta original
        complexity_info: Información de complejidad de la consulta
    
    Returns:
        List[SubQuery]: Lista de subconsultas generadas
    """
    logger.info("Descomponiendo consulta compleja")
    
    # Si la consulta no requiere descomposición, devolvemos una lista vacía
    if not complexity_info.get("requires_decomposition", False):
        logger.info("La consulta no requiere descomposición")
        return []
    
    try:
        # Utilizar el modelo para descomponer la consulta
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Eres un experto en descomposición de consultas complejas sobre normativas y regulaciones de cualquier sector e industria.

Tu tarea es descomponer una consulta compleja en subconsultas más simples y manejables, de forma que:
1. Cada subconsulta aborde un aspecto específico de la consulta original
2. El conjunto de subconsultas cubra completamente la intención original
3. Las subconsultas sean independientes pero relacionadas entre sí

Devuelve un JSON con formato:
{
  "sub_queries": [
    {
      "text": "Texto de la subconsulta",
      "intent": "Intención específica de esta subconsulta",
      "focus": "Aspecto o tema central de esta subconsulta",
      "requires_information_from": ["Otras subconsultas de las que depende, si aplica"]
    },
    ...
  ]
}

Limita la lista a un máximo de 5 subconsultas.
"""},
                {"role": "user", "content": f"Consulta original: {query}\n\nInformación de complejidad: {json.dumps(complexity_info, ensure_ascii=False)}"}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        sub_queries_data = result_json.get("sub_queries", [])
        
        # Convertir a objetos SubQuery
        sub_queries = [SubQuery(**sq_data) for sq_data in sub_queries_data]
        
        logger.info(f"Consulta descompuesta en {len(sub_queries)} subconsultas")
        for i, sq in enumerate(sub_queries):
            logger.info(f"  Subconsulta {i+1}: {sq.text[:100]}..." if len(sq.text) > 100 else sq.text)
        
        return sub_queries
        
    except Exception as e:
        logger.error(f"Error al descomponer consulta: {e}")
        return []

@query_understanding_agent.tool
async def generate_search_query(ctx: RunContext[QueryUnderstandingDeps], original_query: str, expanded_query: str, keywords: List[Keyword], entities: List[Entity]) -> str:
    """
    Genera una consulta optimizada para la búsqueda en la base de conocimientos.
    
    Args:
        original_query: Consulta original
        expanded_query: Consulta expandida
        keywords: Palabras clave extraídas
        entities: Entidades detectadas
    
    Returns:
        str: Consulta optimizada para búsqueda
    """
    logger.info("Generando consulta optimizada para búsqueda")
    
    try:
        # Preparar el contexto para la generación
        keywords_str = ", ".join([k.word for k in keywords])
        entities_str = ", ".join([f"{e.type}:{e.value}" for e in entities])
        
        # Utilizar el modelo para generar la consulta de búsqueda
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.2,
            messages=[
                {"role": "system", "content": """Eres un experto en optimización de consultas para sistemas de recuperación de información vectorial y léxica.

Tu tarea es generar una consulta optimizada que maximice la relevancia en la recuperación de documentos sobre normativas y regulaciones de cualquier sector e industria.. La consulta debe:

1. Incluir términos clave que mejoren tanto la búsqueda semántica (vectorial) como léxica (BM25)
2. Ser concisa pero completa, capturando todos los aspectos importantes
3. Priorizar términos técnicos y específicos del dominio
4. Mantener el balance entre precisión (términos exactos) y recall (cobertura amplia)
5. Evitar términos genéricos que no aportan valor discriminativo

La consulta generada será utilizada para crear embeddings y buscar documentos similares en una base de conocimiento vectorial, así como para búsquedas léxicas.
"""},
                {"role": "user", "content": f"""Consulta original: {original_query}

Consulta expandida: {expanded_query}

Palabras clave identificadas: {keywords_str}

Entidades detectadas: {entities_str}

Genera una consulta optimizada para búsqueda que mantenga la intención original pero maximice la recuperación de documentos relevantes."""}
            ]
        )
        
        search_query = completion.choices[0].message.content
        logger.info(f"Consulta de búsqueda generada: {search_query[:100]}..." if len(search_query) > 100 else search_query)
        
        return search_query
        
    except Exception as e:
        logger.error(f"Error al generar consulta de búsqueda: {e}")
        # Si hay error, devolvemos la consulta expandida o la original
        return expanded_query or original_query

@query_understanding_agent.tool
async def analyze_query(ctx: RunContext[QueryUnderstandingDeps], query: str) -> QueryInfo:
    """
    Realiza un análisis completo de la consulta del usuario.
    
    Args:
        query: Consulta del usuario a analizar
    
    Returns:
        QueryInfo: Información estructurada sobre la consulta
    """
    start_time = time.time()
    logger.info(f"Iniciando análisis completo de la consulta: {query[:100]}..." if len(query) > 100 else query)
    
    # Inicializar estructura de resultado
    query_info = QueryInfo(original_query=query)
    
    try:
        # 1. Detectar idioma
        language_info = await detect_language(ctx, query)
        query_info.language = language_info.get("language_code", "es")
        
        # 2. Traducir si es necesario
        effective_query = query
        if language_info.get("needs_translation", False):
            logger.info(f"La consulta está en {language_info.get('language_name')} y requiere traducción")
            effective_query = await translate_query(ctx, query, language_info.get("language_code"), "es")
            # Guardar consulta original y traducida
            query_info.metadata["original_language"] = language_info.get("language_code")
            query_info.metadata["translated_from"] = query
            query_info.original_query = effective_query
        
        # 3. Extraer entidades
        entities = await extract_entities(ctx, effective_query)
        query_info.entities = entities
        
        # 4. Identificar términos técnicos
        domain_terms = await identify_technical_terms(ctx, effective_query)
        query_info.domain_terms = domain_terms
        
        # 5. Analizar intención
        intents = await analyze_query_intent(ctx, effective_query)
        query_info.intents = intents
        
        # 6. Extraer palabras clave
        keywords = await extract_keywords(ctx, effective_query)
        query_info.keywords = keywords
        
        # 7. Expandir consulta
        expanded_query = await expand_query(ctx, effective_query, entities, domain_terms)
        query_info.expanded_query = expanded_query
        
        # 8. Evaluar complejidad
        complexity_info = await evaluate_complexity(ctx, effective_query, entities)
        query_info.complexity = complexity_info.get("complexity", "simple")
        query_info.metadata["complexity_info"] = complexity_info
        
        # 9. Descomponer consulta si es necesario
        if complexity_info.get("requires_decomposition", False):
            sub_queries = await decompose_query(ctx, effective_query, complexity_info)
            # Convertir las SubQuery a strings para el campo decomposed_queries
            query_info.decomposed_queries = sub_queries
        
        # 10. Generar consulta optimizada para búsqueda
        search_query = await generate_search_query(ctx, effective_query, expanded_query, keywords, entities)
        query_info.search_query = search_query
        
        # 11. Estimar calidad de búsqueda esperada
        query_info.estimated_search_quality = 1.0 - complexity_info.get("estimated_search_difficulty", 0.5)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Análisis de consulta completado en {elapsed_time:.2f}s")
        
        return query_info
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error en el análisis de consulta después de {elapsed_time:.2f}s: {e}")
        
        # Asegurar que devolvemos una estructura mínima válida
        return query_info


async def evaluate_query_complexity(query: str, openai_client) -> bool:
    """
    Evalúa rápidamente si una consulta es compleja y necesita procesamiento avanzado.
    
    Args:
        query: Consulta a evaluar
        openai_client: Cliente de OpenAI
        
    Returns:
        bool: True si la consulta es compleja, False en caso contrario
    """
    logger.info("Evaluando complejidad básica de la consulta")
    
    try:
        # Método simple: evaluación basada en longitud y características básicas
        words = query.split() 
        if len(words) > 15 or query.count('?') > 1:
            logger.info("La consulta se considera compleja por criterios básicos")
            return True
            
        # Evaluación rápida con el modelo
        completion = await openai_client.chat.completions.create(
            model=llm,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Evalúa si la siguiente consulta sobre normativas y regulaciones de cualquier sector e industria.
                Una consulta compleja contiene múltiples preguntas, abarca varios temas, o requiere información de diversas fuentes.
                Responde con un JSON simple: {"is_complex": true/false}"""},
                {"role": "user", "content": query}
            ]
        )
        
        result = json.loads(completion.choices[0].message.content)
        is_complex = result.get("is_complex", False)
        
        logger.info(f"Evaluación rápida de complejidad: {'compleja' if is_complex else 'simple'}")
        return is_complex
        
    except Exception as e:
        logger.error(f"Error evaluando complejidad básica: {str(e)}")
        # Por defecto, asumimos que no es compleja en caso de error
        return False

async def extract_entities_fast(openai_client, text: str) -> List[Entity]:
    """
    Versión rápida y simplificada de extracción de entidades.
    """
    try:
        completion = await openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extrae las principales entidades (regulation, process, card_type, fee, region) de la consulta. Responde con formato JSON: {\"entities\": [{\"type\": \"...\", \"value\": \"...\"}]}"},
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        entities_data = result_json.get("entities", [])
        
        return [Entity(**entity_data) for entity_data in entities_data]
        
    except Exception as e:
        logger.error(f"Error en extracción rápida de entidades: {e}")
        return []

async def extract_keywords_fast(openai_client, text: str) -> List[Keyword]:
    """
    Versión rápida y simplificada de extracción de palabras clave.
    """
    try:
        completion = await openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extrae las 3-5 palabras clave más importantes de la consulta. Responde con formato JSON: {\"keywords\": [{\"word\": \"...\", \"importance\": 0.9}]}"},
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        keywords_data = result_json.get("keywords", [])
        
        return [Keyword(**keyword_data) for keyword_data in keywords_data]
        
    except Exception as e:
        logger.error(f"Error en extracción rápida de palabras clave: {e}")
        return []

async def analyze_intent_fast(openai_client, text: str) -> List[Intent]:
    """
    Versión rápida y simplificada de análisis de intención.
    """
    try:
        completion = await openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Identifica la intención principal de la consulta sobre normativas y regulaciones de cualquier sector e industria. Responde con formato JSON: {\"intents\": [{\"name\": \"...\", \"confidence\": 0.9}]}"},
                {"role": "user", "content": text}
            ]
        )
        
        result_json = json.loads(completion.choices[0].message.content)
        intents_data = result_json.get("intents", [])
        
        return [Intent(**intent_data) for intent_data in intents_data]
        
    except Exception as e:
        logger.error(f"Error en análisis rápido de intención: {e}")
        return [Intent(name="consulta_general", confidence=0.5)]

async def generate_search_query_fast(openai_client, query: str, keywords: List[Keyword], entities: List[Entity]) -> str:
    """
    Versión rápida para generar una consulta de búsqueda.
    """
    try:
        keywords_str = ", ".join([k.word for k in keywords])
        entities_str = ", ".join([f"{e.type}:{e.value}" for e in entities])
        
        completion = await openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Reformula esta consulta para optimizarla para búsqueda vectorial y léxica, priorizando términos técnicos y específicos."},
                {"role": "user", "content": f"Consulta: {query}\nPalabras clave: {keywords_str}\nEntidades: {entities_str}"}
            ]
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generando consulta de búsqueda rápida: {e}")
        return query

async def process_query(query: str, deps: QueryUnderstandingDeps) -> QueryInfo:
    """
    Procesa una consulta y aplica las técnicas de comprensión avanzada de manera adaptativa.
    
    Args:
        query: La consulta del usuario
        deps: Las dependencias necesarias para el agente
    
    Returns:
        QueryInfo: Información estructurada sobre la consulta
    """
    start_time = time.time()
    logger.info(f"Procesando consulta con el agente de comprensión: {query[:100]}..." if len(query) > 100 else query)
    
    # Inicializar estructura de resultado
    query_info = QueryInfo(original_query=query)
    
    try:
        # En lugar de crear un RunContext, utilizaremos directamente el agente
        # con la función run que ejecutará todas las herramientas necesarias
        
        # 1. Detectar idioma (simple, sin usar RunContext)
        logger.info("Detectando idioma del texto")
        try:
            # Enfoque directo sin RunContext
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """Eres un experto en detección de idiomas. 
                    
    Tu tarea es identificar el idioma en que está escrito el texto proporcionado.
    
    Devuelve un JSON con formato:
    {
      "language_code": "código_iso_639_1",
      "language_name": "nombre del idioma en español",
      "confidence": 0.95,
      "needs_translation": true/false,
      "detected_script": "latino/cirílico/etc"
    }
    
    El valor de 'needs_translation' debe ser true si el idioma NO es español, y false si es español.
    """},
                    {"role": "user", "content": query}
                ]
            )
            
            language_info = json.loads(completion.choices[0].message.content)
            query_info.language = language_info.get("language_code", "es")
            logger.info(f"Idioma detectado: {language_info.get('language_name', 'desconocido')} ({language_info.get('language_code', '??')})")
        except Exception as e:
            logger.error(f"Error en la detección de idioma: {str(e)}")
            query_info.language = "es"  # Default a español en caso de error
        
        # 2. Evaluación de complejidad simple
        # Corregir el orden de los argumentos
        is_complex = await evaluate_query_complexity(query, deps.openai_client)
        processing_level = "full" if is_complex else "basic"
        
        logger.info(f"Nivel de procesamiento seleccionado: {processing_level}")
        
        # 3. Procesamiento adaptativo basado en complejidad
        if processing_level == "basic":
            # Para consultas simples, hacemos solo procesamiento directo sin usar herramientas del agente
            
            # 3.1 Extracción de entidades
            try:
                completion = await deps.openai_client.chat.completions.create(
                    model=llm,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": """Extrae las principales entidades (regulation, process, card_type, fee, region) de la consulta. Responde con formato JSON: {\"entities\": [{\"type\": \"...\", \"value\": \"...\"}]}"""},
                        {"role": "user", "content": query}
                    ]
                )
                
                result_json = json.loads(completion.choices[0].message.content)
                entities_data = result_json.get("entities", [])
                entities = [Entity(**entity_data) for entity_data in entities_data]
                query_info.entities = entities
            except Exception as e:
                logger.error(f"Error en extracción de entidades: {e}")
                query_info.entities = []
            
            # 3.2 Extracción de palabras clave
            try:
                completion = await deps.openai_client.chat.completions.create(
                    model=llm,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": """Extrae las 3-5 palabras clave más importantes de la consulta. Responde con formato JSON: {\"keywords\": [{\"word\": \"...\", \"importance\": 0.9}]}"""},
                        {"role": "user", "content": query}
                    ]
                )
                
                result_json = json.loads(completion.choices[0].message.content)
                keywords_data = result_json.get("keywords", [])
                keywords = [Keyword(**keyword_data) for keyword_data in keywords_data]
                query_info.keywords = keywords
            except Exception as e:
                logger.error(f"Error en extracción de palabras clave: {e}")
                query_info.keywords = []
            
            # 3.3 Análisis de intención
            try:
                completion = await deps.openai_client.chat.completions.create(
                    model=llm,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": """Identifica la intención principal de la consulta sobre normativas y regulaciones de cualquier sector e industria. Responde con formato JSON: {\"intents\": [{\"name\": \"...\", \"confidence\": 0.9}]}"""},
                        {"role": "user", "content": query}
                    ]
                )
                
                result_json = json.loads(completion.choices[0].message.content)
                intents_data = result_json.get("intents", [])
                intents = [Intent(**intent_data) for intent_data in intents_data]
                query_info.intents = intents
            except Exception as e:
                logger.error(f"Error en análisis de intención: {e}")
                query_info.intents = [Intent(name="consulta_general", confidence=0.5)]
            
            # 3.4 Generación de consulta de búsqueda
            query_info.complexity = "simple"
            query_info.search_query = query  # Usar la consulta original como predeterminada
            
            try:
                keywords_str = ", ".join([k.word for k in query_info.keywords])
                entities_str = ", ".join([f"{e.type}:{e.value}" for e in query_info.entities])
                
                completion = await deps.openai_client.chat.completions.create(
                    model=llm,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "Reformula esta consulta para optimizarla para búsqueda vectorial y léxica, priorizando términos técnicos y específicos."},
                        {"role": "user", "content": f"Consulta: {query}\nPalabras clave: {keywords_str}\nEntidades: {entities_str}"}
                    ]
                )
                
                query_info.search_query = completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generando consulta de búsqueda: {e}")
                # Mantener la consulta original si hay error
            
        else:
            # Para consultas complejas, utilizaremos el método execute del agente
            # que no depende de RunContext
            logger.info("Utilizando el agente completo para procesar la consulta")
            try:
                # En lugar de usar execute directamente, vamos a llamar a cada herramienta individualmente
                result = await query_understanding_agent.run(
                    f"Analiza esta consulta completamente: {query}",
                    deps=deps
                )
                # Procesar resultado si es necesario
                # Si el agente devuelve un QueryInfo completo, podemos utilizarlo directamente
                if isinstance(result, QueryInfo):
                    query_info = result
                else:
                    # En caso contrario, mantener la información básica ya recopilada
                    logger.info("El agente no devolvió un QueryInfo completo, usando información parcial")
            except Exception as e:
                logger.error(f"Error ejecutando el agente de comprensión: {e}")
                # Mantener la información básica ya recopilada
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error en el procesamiento de la consulta después de {elapsed_time:.2f}s: {e}")
        # Asegurar que devolvemos un resultado básico pero funcional
        query_info.metadata["error"] = str(e)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Tiempo total de procesamiento de la consulta: {elapsed_time:.2f}s")
    
    return query_info