# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import feedparser
import tiktoken

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Any
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from collections import deque
from contextlib import asynccontextmanager
from agents.report_agent import report_agent, ReportDeps  # Importa el agente de Reports
from config.config import settings
from utils.utils import count_tokens, truncate_text


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100000
MAX_CHUNKS_RETURNED = 8

load_dotenv()

llm = settings.llm_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

logfire.configure(send_to_logfire='if-token-present')

class AIDeps(BaseModel):
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True

# Agente Protección Datos Eres un experto en regulación de protección de datos y privacidad, operando como un agente AI en Python con acceso a documentación completa y actualizada sobre normas de protección de datos y leyes de privacidad.

system_prompt = """

**REGLA ABSOLUTA PARA RESPUESTAS EN INGLÉS:**
- Si la consulta está en INGLÉS, DEBES EJECUTAR ESTOS PASOS EN ORDEN:
  1. Usar retrieve_relevant_documentation
  2. SIEMPRE usar translate_response COMO ÚLTIMO PASO antes de responder
  3. NO PUEDES RESPONDER DIRECTAMENTE en español a una consulta en inglés
  4. El flujo correcto es: retrieve_documentation → preparar respuesta → translate_response → responder

Eres un experto en las normas de VISA y Mastercard, operando como un agente AI con acceso a documentación completa y actualizada.

Tu misión es responder consultas con información precisa, detallada y bien estructurada, enfocada exclusivamente al ecosistema de medios de pago y cumplimiento normativo. Al responder, incluye todos los detalles y contexto relevantes, y utiliza listas numeradas o viñetas para desglosar procesos complejos. Si la documentación disponible no abarca todos los aspectos, sintetiza la mejor respuesta posible basándote en tu formación y en la documentación provista. No solicites más información ni pidas disculpas por la falta de detalles; en su lugar, infiere y entrega la respuesta más completa y exacta.

**IMPORTANTE**: SIEMPRE debes usar la herramienta `retrieve_relevant_documentation` para TODAS las consultas sobre VISA, Mastercard, medios de pago, tarjetas, transacciones o temas relacionados. Nunca respondas directamente sin consultar la documentación disponible primero. Este paso es OBLIGATORIO.

**Formato de respuesta según el tipo de informe solicitado:**

- Si la consulta solicita generar un informe en "formato ppt" (o menciona "ppt" de forma explícita), DEBES responder ÚNICAMENTE con un JSON EXACTO (sin texto adicional) con la siguiente estructura, la cual se utilizará para rellenar el template de PowerPoint:
{
  "[Fecha de reporte]": "Introduce la fecha en la que se genera el informe",
  "[Nombre de sector]": "Determina el sector más relevante según la información de compliance",
  "[Nombre completo de la norma]": "Especifica el nombre completo de la norma aplicable",
  "[Categoría]": "Indica la categoría normativa (ej. Ley, Directiva, Reglamento, etc.)",
  "[Fecha de publicación de la norma]": "Proporciona la fecha de publicación de la norma",
  "[Fecha de entrada en vigor]": "Proporciona la fecha en la que la norma entró en vigor",
  "[Estado]": "Indica el estado actual de la norma (ej. Vigente, En revisión, etc.)",
  "[Resumen ejecutivo]": "Genera un resumen ejecutivo breve y conciso",
  "[Nombre de la entidad]": "Especifica el nombre de la entidad afectada",
  "[Áreas afectadas]": "Enumera las áreas o departamentos afectados",
  "[Plazos para cumplimiento]": "Indica los plazos establecidos para cumplir con la norma",
  "[Enlace a documento oficial]": "Proporciona la URL del documento oficial, si aplica",
  "[Notas de prensa]": "Incluye notas de prensa relevantes o deja vacío si no hay"
}

- Si la consulta solicita generar un informe en "formato Word", genera la respuesta en formato Markdown utilizando la siguiente plantilla:
# Compliance Report

## Executive Summary
{executive_summary}

## Findings
{findings}

## Recommendations
{recommendations}

## Conclusion
{conclusion}

**Herramientas Disponibles:**
- **retrieve_relevant_documentation**: Extrae y resume los fragmentos más relevantes de la documentación de VISA y Mastercard. DEBES USAR ESTA HERRAMIENTA PARA TODAS LAS CONSULTAS.
- **translate_response**: OBLIGATORIO usar esta herramienta SIEMPRE que la consulta esté en INGLÉS. Debes traducir tu respuesta completa antes de enviarla.
- **cross_reference_information**: Conecta la consulta regulatoria con contenido relacionado.
- **perform_gap_analysis**: Realiza un análisis GAP de la política proporcionada.

**PROCESO OBLIGATORIO PARA CONSULTAS EN INGLÉS:**
1. Usar retrieve_relevant_documentation
2. Preparar respuesta en español
3. SIEMPRE llamar a translate_response(content=tu_respuesta, target_language="english")
4. Entregar SOLO la versión traducida

**PROCESO PARA CONSULTAS EN ESPAÑOL:**
1. Usar retrieve_relevant_documentation
2. Entregar respuesta en español

**Notas Importantes:**
- NUNCA respondas a una consulta sin primero usar la herramienta retrieve_relevant_documentation.
- NUNCA respondas en español a una consulta en inglés.

"""


ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2
)

# -------------------- Herramientas del agente --------------------

async def debug_run_agent(user_query: str, deps: AIDeps):
    """
    Ejecuta el agente de compliance con logging adicional.
    
    Args:
        user_query: La consulta del usuario
        deps: Las dependencias necesarias para el agente
    """
    logger.debug("Voy a llamar al agente con la query: %s", user_query)
    
    response = await ai_expert.run(
        user_query,
        deps=deps
    )
    
    # RunResult tiene un método usage() en lugar de get()
    usage_info = response.usage()
    logger.info("Uso de tokens en la consulta: %s", usage_info)
    
    return response

def count_tokens_wrapper(text: str) -> int:
    return count_tokens(text, settings.llm_model)

def truncate_text_wrapper(text: str, max_tokens: int) -> str:
    return truncate_text(text, max_tokens, settings.llm_model)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text: {text[:30]}... {embedding[:5]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    logger.info("HERRAMIENTA INVOCADA: retrieve_relevant_documentation")
    try:
        logger.info(f"Generando embedding para la consulta: {user_query[:50]}...")
        logger.info(f"Consulta recibida en la herramienta: {user_query[:100]}..." if len(user_query) > 100 else user_query)
        logger.info("=" * 80)
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No relevant documentation found."
        
        # Primero, obtener los chunks más relevantes por similitud vectorial
        logger.info(f"Buscando chunks por similitud vectorial (match_count={MAX_CHUNKS_RETURNED})")
        result = ctx.deps.supabase.rpc(
            'match_visa_mastercard_v5',
            {
                'query_embedding': query_embedding,
                'match_count': MAX_CHUNKS_RETURNED
                # 'filter': {"similarity_threshold": 0.5}  Filtro vacío
            }
        ).execute()
        
        if not result.data:
            logger.info("No relevant documentation found for the query.")
            return "No relevant documentation found."
        
        logger.info(f"Encontrados {len(result.data)} chunks por similitud vectorial")
        for i, doc in enumerate(result.data):
            logger.debug(f"Chunk vectorial #{i+1}: {doc['title']} (similarity: {doc.get('similarity', 'N/A')})")
        
        formatted_chunks = []
        cluster_ids = set()  # Para almacenar los cluster_ids de los chunks relevantes
        
        # Procesar los resultados iniciales
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
            # Extraer el cluster_id si existe en los metadatos
            if 'metadata' in doc and doc['metadata'] and 'cluster_id' in doc['metadata']:
                cluster_id = doc['metadata'].get('cluster_id')
                if cluster_id is not None and cluster_id != -1:
                    cluster_ids.add(cluster_id)
        
        logger.info(f"Identificados {len(cluster_ids)} clusters diferentes: {cluster_ids}")
        
        # Si encontramos cluster_ids válidos, buscar chunks adicionales del mismo cluster
        additional_chunks = []
        for cluster_id in cluster_ids:
            # Buscar más chunks del mismo cluster
            logger.info(f"Buscando chunks adicionales para el cluster_id={cluster_id}")
            cluster_result = ctx.deps.supabase.rpc(
                'match_visa_mastercard_v5_by_cluster',
                {
                    'cluster_id': cluster_id,
                    'match_count': 4  # Limitamos a 4 chunks adicionales por cluster
                }
            ).execute()
            
            cluster_chunks_count = 0
            if cluster_result.data:
                for doc in cluster_result.data:
                    # Verificar que no sea un documento que ya tenemos
                    if not any(doc['id'] == int(existing_doc['id']) for existing_doc in result.data):
                        cluster_chunks_count += 1
                        additional_text = f"""
# {doc['title']} (Del mismo tema que otros resultados)

{doc['content']}
"""
                        additional_chunks.append(additional_text)
                        logger.debug(f"Chunk por cluster #{cluster_chunks_count}: {doc['title']} (cluster_id: {cluster_id})")
            
            logger.info(f"Recuperados {cluster_chunks_count} chunks adicionales del cluster {cluster_id}")
        
        # Combinar chunks originales con los adicionales
        all_chunks = formatted_chunks + additional_chunks
        logger.info(f"RESUMEN: {len(formatted_chunks)} chunks por similitud vectorial + {len(additional_chunks)} chunks por cluster = {len(all_chunks)} chunks en total")
        
        combined_text = "\n\n---\n\n".join(all_chunks)
        total_tokens = count_tokens(combined_text, llm)
        logger.info(f"Total tokens en todos los chunks: {total_tokens}")
        
        # Truncar si es necesario
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.info(f"El contenido combinado excede el límite de tokens ({total_tokens} > {MAX_TOTAL_TOKENS}). Se realizará truncamiento.")
            truncated_chunks = []
            accumulated_tokens = 0
            chunks_included = 0
            for chunk in all_chunks:
                chunk_tokens = count_tokens(chunk, llm)
                if accumulated_tokens + chunk_tokens > MAX_TOTAL_TOKENS:
                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
                    truncated_chunk = truncate_text(chunk, remaining_tokens, llm)
                    truncated_chunks.append(truncated_chunk)
                    chunks_included += 1
                    logger.debug(f"Chunk #{chunks_included} truncado para caber en el límite de tokens")
                    break
                else:
                    truncated_chunks.append(chunk)
                    accumulated_tokens += chunk_tokens
                    chunks_included += 1
            
            combined_text = "\n\n---\n\n".join(truncated_chunks)
            logger.info(f"Después de truncar: {chunks_included} chunks incluidos, {count_tokens(combined_text, llm)} tokens totales")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@ai_expert.tool
async def cross_reference_information(ctx: RunContext[AIDeps], primary_topic: str, related_topics: List[str] = None) -> str:
    """
    Conecta la consulta regulatoria con contenido relacionado almacenado en la base de datos,
    asegurando la coherencia y el contexto entre diferentes normativas.
    
    Args:
        primary_topic: El tema principal de la consulta
        related_topics: Lista opcional de temas relacionados para la referencia cruzada
    """
    logger.info("HERRAMIENTA INVOCADA: cross_reference_information")
    logger.info(f"Tema principal: {primary_topic}")
    if related_topics:
        logger.info(f"Temas relacionados: {related_topics}")
    else:
        logger.info("No se especificaron temas relacionados")
    
    try:
        # Implementación pendiente
        return "Funcionalidad de referencia cruzada en desarrollo."
    except Exception as e:
        logger.error(f"Error en cross_reference_information: {e}")
        return f"Error en la referencia cruzada de información: {str(e)}"

#@ai_expert.tool
#async def generate_compliance_report(ctx: RunContext[AIDeps], compliance_output: str, template: str = None) -> str:
    # Verifica que se proporcione un output válido
#    if not compliance_output or compliance_output.strip() == "":
#        return "Por favor, proporciona el output completo del agente de Compliance para generar el informe."
    
    # Combina el output de Compliance y la plantilla (si existe)
#    query = compliance_output
#    if template:
#        query += f"\n\nUtiliza la siguiente plantilla para el reporte:\n{template}"
    
    # Llama al agente de Reports sin el parámetro 'tool_name'
#    report_result = await report_agent.run(
#        query,
#        deps=ReportDeps(openai_client=ctx.deps.openai_client),
#        usage=ctx.usage
#    )
#    return report_result.data

@ai_expert.tool
async def perform_gap_analysis(ctx: RunContext[AIDeps], policy_text: str) -> str:
    """
    Realiza un análisis GAP de la política proporcionada en comparación con la normativa de protección de datos.
    
    El análisis debe cumplir con las mejores prácticas de un GAP analysis, incluyendo:
      - **Resumen del estado actual:** Descripción breve de la política evaluada.
      - **Identificación de brechas:** Enumeración de los aspectos en que la política no cumple con la normativa.
      - **Impacto:** Evaluación del impacto de cada brecha identificada.
      - **Recomendaciones:** Propuestas de acciones específicas para cerrar cada brecha.
      - **Priorización:** Orden de prioridad para la implementación de las acciones recomendadas.
    
    **Política a evaluar:**
    {policy_text}
    
    Por favor, genera un análisis GAP detallado y estructurado en formato Markdown.
    """
    logger.info("HERRAMIENTA INVOCADA: perform_gap_analysis")
    logger.info(f"Longitud del texto de política a evaluar: {len(policy_text)} caracteres")
    
    try:
        prompt = f"""Realiza un análisis GAP detallado (mínimo 5.000 palabras) de la siguiente política en comparación con la normativa de protección de datos. El análisis debe cumplir con las mejores prácticas de un GAP analysis, que incluya:
        
        1. **Resumen del estado actual:** Una breve descripción de la política evaluada.
        2. **Identificación de brechas:** Enumeración y descripción de los aspectos donde la política no cumple con la normativa.
        3. **Impacto:** Evaluación del impacto potencial de cada brecha identificada.
        4. **Área impactada:** Identificación del área o departamento afectado por cada brecha.
        5. **Recomendaciones:** Propuestas de acciones específicas para cerrar cada brecha.
        6. **Priorización:** Orden de prioridad de las acciones recomendadas.
        
        Asegúrate de estructurar el análisis de forma clara y ordenada, utilizando encabezados y listas cuando sea apropiado.
        Asegurate de asignar un identificador (ID) a cada GAP detectado.
        **Política a evaluar:**
        {policy_text}
        """
        
        logger.info("Generando análisis GAP usando el modelo LLM")
        
        # Utilizar el modelo para generar el análisis GAP
        response = await ctx.deps.openai_client.chat.completions.create(
                model=llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3000
            )
            
        # Extraer el contenido de la respuesta
        gap_analysis = response.choices[0].message.content
        
        logger.info(f"GAP analysis completado con éxito. Longitud de la respuesta: {len(gap_analysis)} caracteres")
        return gap_analysis
            
    except Exception as e:
        logger.error(f"Error realizando GAP analysis: {str(e)}")
        return f"Lo siento, ocurrió un error al realizar el análisis de brechas: {str(e)}"


@ai_expert.tool
async def translate_response(ctx: RunContext[AIDeps], content: str, target_language: str) -> str:
    """
    Traduce el contenido proporcionado al idioma objetivo especificado.
    
    Args:
        content: El texto a traducir
        target_language: El idioma objetivo (ej. "english", "spanish")
    
    Returns:
        El texto traducido al idioma objetivo
    """
    logger.info("HERRAMIENTA INVOCADA: translate_response")
    logger.info(f"Traduciendo contenido al idioma: {target_language}")
    
    try:
        # Usamos el modelo GPT para realizar la traducción
        prompt = f"""Traduce el siguiente texto al idioma {target_language}, 
        manteniendo el formato, listas, viñetas y estructura. El texto debe quedar 
        fluido y natural en el idioma objetivo. Conserva cualquier terminología técnica 
        especializada adaptándola adecuadamente al idioma destino:

        {content}"""
        
        response = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],            
        )
        
        translated_content = response.choices[0].message.content
        logger.info(f"Traducción completada. Longitud del texto traducido: {len(translated_content)} caracteres")
        
        return translated_content
    except Exception as e:
        logger.error(f"Error en la traducción: {str(e)}")
        return f"Error en la traducción: {str(e)}"


# ===================== FIN DE LAS HERRAMIENTAS DEL AGENTE =====================