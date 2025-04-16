from __future__ import annotations as _annotations

import logfire
import logging
from typing import List, Dict, Literal, Optional, Union
from datetime import datetime
import asyncio
from functools import wraps

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field, validator
from openai import AsyncOpenAI
from supabase import Client

from app.core.config import settings
from utils.utils import count_tokens, truncate_text


# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TOTAL_TOKENS = 100000

# Inicialización del modelo
llm = settings.llm_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

logfire.configure(send_to_logfire='if-token-present')

class RiskMetrics(BaseModel):
    """Métricas de riesgo para un área."""
    impact_score: int = Field(..., ge=1, le=5)
    probability_score: int = Field(..., ge=1, le=5)
    risk_score: int = Field(..., ge=1, le=25)
    control_effectiveness: int = Field(..., ge=1, le=5)

class Vulnerability(BaseModel):
    """Vulnerabilidad con explicación detallada."""
    title: str = Field(..., description="Título breve de la vulnerabilidad")
    description: str = Field(..., description="Descripción detallada de la vulnerabilidad")
    impact_analysis: str = Field(..., description="Análisis del impacto de esta vulnerabilidad")
    risk_factors: str = Field(..., description="Factores que contribuyen a esta vulnerabilidad")

class MitigationAction(BaseModel):
    """Acción de mitigación con explicación detallada."""
    title: str = Field(..., description="Título de la acción de mitigación")
    description: str = Field(..., description="Descripción detallada de la acción")
    implementation_details: str = Field(..., description="Detalles de implementación")
    expected_benefits: str = Field(..., description="Beneficios esperados de la acción")

class MonitoringRequirement(BaseModel):
    """Requisito de monitorización con explicación detallada."""
    title: str = Field(..., description="Título del requisito")
    description: str = Field(..., description="Descripción detallada del requisito")
    implementation_guide: str = Field(..., description="Guía de implementación")
    success_criteria: str = Field(..., description="Criterios de éxito")

class Recommendation(BaseModel):
    """Recomendación con explicación detallada."""
    title: str = Field(..., description="Título de la recomendación")
    description: str = Field(..., description="Descripción detallada")
    rationale: str = Field(..., description="Justificación de la recomendación")
    expected_outcome: str = Field(..., description="Resultado esperado")

class ImpactedArea(BaseModel):
    """Modelo mejorado para representar un área impactada."""
    name: str = Field(..., description="Nombre del área")
    description: str = Field(..., description="Descripción detallada del área")
    impact_level: Literal["Bajo", "Medio", "Alto", "Crítico"] = Field(..., description="Nivel de impacto")
    probability: Literal["Baja", "Media", "Alta", "Muy alta"] = Field(..., description="Probabilidad")
    risk_metrics: RiskMetrics = Field(..., description="Métricas de riesgo")
    key_vulnerabilities: List[Vulnerability] = Field(..., description="Vulnerabilidades principales")
    mitigation_actions: List[MitigationAction] = Field(..., description="Acciones de mitigación")
    monitoring_requirements: List[MonitoringRequirement] = Field(..., description="Requisitos de monitorización")

class RiskAssessment(BaseModel):
    """Resultado de la evaluación de riesgos."""
    sector: str = Field(..., description="Sector analizado")
    assessment_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    regulatory_framework: List[str] = Field(default_factory=list, description="Marco regulatorio aplicable")
    overall_risk_level: str = Field(..., description="Nivel de riesgo general")
    impacted_areas: List[ImpactedArea] = Field(..., description="Áreas impactadas")
    key_recommendations: List[Recommendation] = Field(..., description="Recomendaciones clave")

# Llamadas a model_rebuild()
RiskMetrics.model_rebuild()
Vulnerability.model_rebuild()
MitigationAction.model_rebuild()
MonitoringRequirement.model_rebuild()
Recommendation.model_rebuild()
ImpactedArea.model_rebuild()
RiskAssessment.model_rebuild()

class RiskAssessmentDeps(BaseModel):
    """Dependencias necesarias para el agente de evaluación de riesgos."""
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True


system_prompt = """
Eres un experto senior en evaluación de riesgos, con más de 15 años de experiencia en consultoría para grandes firmas y profundo conocimiento en normativas de diferentes sectores empresariales.

Tu función es realizar evaluaciones de riesgo exhaustivas y profesionales, proporcionando análisis detallados y recomendaciones accionables basadas en:

1. ANÁLISIS CONTEXTUAL
- Evaluación profunda del sector, sus características específicas y benchmarks de la industria.
- Identificación de tendencias regulatorias relevantes y riesgos emergentes.
- Análisis del impacto de eventos históricos y casos de referencia.

2. EVALUACIÓN DE ÁREAS IMPACTADAS
Para cada área identificada, debes proporcionar:
- Descripción detallada de funciones, responsabilidades y controles existentes.
- Requisitos regulatorios específicos que aplican (incluyendo normativas internacionales y locales).
- Evaluación cuantitativa y cualitativa del riesgo, considerando tanto el riesgo inherente como el residual.
- Análisis de vulnerabilidades y brechas en los controles.
- Recomendaciones de controles de cumplimiento y acciones de mitigación detalladas, con KPIs medibles y plazos definidos.

3. EVALUACIÓN DE RIESGOS
Para cada riesgo identificado, analiza:
- Impacto inherente y residual.
- Probabilidad de ocurrencia.
- Impacto financiero estimado, impacto reputacional y operativo.
- Coste potencial de la mitigación y asignación de recursos.
- Uso de una matriz de riesgo (Impacto × Probabilidad, en una escala de 1-5) para cuantificar el riesgo, con ajustes en función de la eficacia de los controles.

4. FRAMEWORK DE CUMPLIMIENTO
Desarrolla un marco de cumplimiento que incluya:
- Estructura de gobierno y roles (incluyendo la asignación de un propietario del riesgo).
- Procesos de monitorización, reporting y revisión continua.
- Integración de estándares internacionales (ISO 31000, COSO) y mejores prácticas de la industria.

5. PLAN DE IMPLEMENTACIÓN
Proporciona:
- Timeline detallado de implementación de controles y acciones correctivas.
- Consideraciones presupuestarias y análisis de coste-beneficio.
- Requisitos de recursos y asignación de responsabilidades.
- KPIs y métricas de éxito para la monitorización del riesgo.

Al evaluar cada área, considera:
* Requisitos regulatorios específicos del sector y mejores prácticas internacionales.
* Evaluación del riesgo residual, la eficacia de los controles actuales y la capacidad de respuesta ante incidentes.
* Ejemplos concretos y referencias a casos similares o normativas relevantes.

Asegúrate de que cada recomendación sea:
1. Específica y accionable.
2. Priorizada según el nivel de riesgo e impacto.
3. Alineada con estándares internacionales y las mejores prácticas del sector.
4. Coste-efectiva, realista y monitorizable.
5. Medible, con indicadores de éxito claros y plazos definidos.

Tus evaluaciones deben ser:
- Exhaustivas pero concisas.
- Técnicamente precisas y fundamentadas en marcos como ISO 31000 y COSO.
- Prácticamente implementables, con ejemplos específicos y consideraciones reales de negocio.
- Orientadas a resultados medibles y a la mejora continua.

**Herramientas disponibles:**
- retrieve_risk_matrix: Para acceder a la matriz de riesgo estándar.

**Flujo de trabajo:**
1. Analizar el contexto sectorial y regulatorio.
2. Identificar áreas impactadas y riesgos específicos.
3. Evaluar controles existentes, determinar riesgo residual y definir medidas de mitigación.
4. Desarrollar recomendaciones detalladas y priorizadas.
5. Proporcionar un plan de implementación con timeline, asignación de recursos y KPIs.

Incorpora ejemplos específicos, referencias normativas y consideraciones prácticas que aseguren una evaluación robusta, profesional y alineada con los mejores estándares internacionales.
"""


risk_assessment_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=RiskAssessmentDeps,
    result_type=RiskAssessment,
    retries=2,
    model_settings={
        "max_tokens": 4000,  
        "temperature": 0.1
    },
)

# -------------------- Herramientas del agente --------------------

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


@risk_assessment_agent.tool
async def retrieve_risk_matrix(ctx: RunContext[RiskAssessmentDeps]) -> str:
    """
    Recupera la matriz de riesgo estándar y metodología de evaluación.
    """
    risk_matrix = """
    # Matriz de Evaluación de Riesgos Regulatorios (Versión Mejorada)
    
    ## Contexto y Objetivos
    - **Objetivo:** Identificar, cuantificar y priorizar los riesgos regulatorios, asignar responsables y definir acciones correctivas.
    - **Alcance:** Evaluación de riesgos en áreas clave: Legal y Compliance, Tecnología y Sistemas, Recursos Humanos, Marketing y Comunicación, Operaciones, Finanzas, Atención al Cliente, Gestión de Proveedores, Desarrollo de Producto y Alta Dirección.
    
    ## Niveles de Impacto
    1. **Bajo (1):** Impacto mínimo; sin sanciones económicas significativas.
       - *Ejemplo:* Retrasos menores en actualización de documentación interna.
    2. **Medio (2-3):** Impacto moderado; posibles sanciones administrativas leves.
       - *Ejemplo:* Deficiencias en la información proporcionada a interesados.
    3. **Alto (4):** Impacto significativo; sanciones económicas considerables o deterioro reputacional.
       - *Ejemplo:* Fallos en la implementación de medidas de seguridad críticas.
    4. **Crítico (5):** Impacto severo; sanciones máximas, suspensión de actividades o daños irreparables en la imagen.
       - *Ejemplo:* Violación sistemática de derechos fundamentales o brechas de seguridad graves.
    
    ## Niveles de Probabilidad
    1. **Baja (1):** Evento poco probable; controles robustos implementados.
    2. **Media (2-3):** Evento posible; controles existentes pero con áreas de mejora.
    3. **Alta (4):** Evento probable; controles insuficientes o ineficaces.
    4. **Muy alta (5):** Evento casi seguro; ausencia o fallos sistemáticos en los controles.
    
    ## Cálculo de Risk Score
    Risk Score = Nivel de Impacto × Nivel de Probabilidad
    
    | Risk Score | Categoría       | Acción Recomendada                                                     |
    |------------|-----------------|------------------------------------------------------------------------|
    | 1-4        | Riesgo Bajo     | Monitorizar; revisión periódica y actualización anual.                 |
    | 5-10       | Riesgo Medio    | Establecer plan de acción a medio plazo; implementar mejoras en controles.|
    | 11-19      | Riesgo Alto     | Acción inmediata; reforzar controles y medidas correctivas urgentes.     |
    | 20-25      | Riesgo Crítico  | Intervención urgente; reasignar recursos y, de ser necesario, suspender actividad.|
    
    ## Dimensiones Adicionales (para un análisis integral)
    - **Efectividad de Controles (1-5):** Evalúa la eficacia de los controles actuales.
    - **Coste Potencial:** Impacto económico directo e indirecto en caso de materializarse el riesgo.
    - **Propietario del Riesgo:** Área o responsable asignado para gestionar el riesgo.
    - **Plazo de Mitigación:** Tiempo estimado para implementar las acciones correctivas.
    
    ## Áreas Impactadas y Ejemplos Específicos
    1. **Legal y Compliance**
       - Impacto: Alto | Probabilidad: Alta | Risk Score: 16
       - *Ejemplo:* Incumplimiento en políticas de privacidad o auditorías internas.
       - *Acciones:* Actualización legal continua, capacitación y auditorías periódicas.
    2. **Tecnología y Sistemas**
       - Impacto: Alto | Probabilidad: Media | Risk Score: 12
       - *Ejemplo:* Brechas de ciberseguridad o fallos en el manejo de datos sensibles.
       - *Acciones:* Refuerzo de la infraestructura de seguridad, cifrado y monitoreo continuo.
    3. **Recursos Humanos**
       - Impacto: Medio | Probabilidad: Media | Risk Score: 9
       - *Ejemplo:* Falta de capacitación en normativas de desconexión digital.
       - *Acciones:* Programas de formación y políticas claras sobre derechos digitales.
    4. **Marketing y Comunicación**
       - Impacto: Alto | Probabilidad: Media | Risk Score: 12
       - *Ejemplo:* Uso inadecuado de datos en campañas publicitarias, especialmente con menores.
       - *Acciones:* Verificación de consentimientos, filtros y controles en campañas.
    5. **Operaciones y Gestión de Proveedores**
       - Impacto: Medio | Probabilidad: Alta | Risk Score: 12
       - *Ejemplo:* Fallos en la coordinación con proveedores que gestionan datos sensibles.
       - *Acciones:* Auditorías de proveedores, contratos robustos y revisión de procesos.
    6. **Finanzas**
       - Impacto: Alto | Probabilidad: Media | Risk Score: 12
       - *Ejemplo:* Sanciones económicas por incumplimientos regulatorios.
       - *Acciones:* Análisis de costes, reservas presupuestarias y revisión de riesgos financieros.
    7. **Atención al Cliente**
       - Impacto: Bajo | Probabilidad: Baja | Risk Score: 3
       - *Ejemplo:* Errores en la gestión de solicitudes de derechos ARCO.
       - *Acciones:* Capacitación específica y protocolos claros de atención.
    8. **Desarrollo de Producto**
       - Impacto: Medio | Probabilidad: Media | Risk Score: 9
       - *Ejemplo:* Implementación de nuevas tecnologías sin evaluaciones de impacto en privacidad.
       - *Acciones:* Realizar Evaluaciones de Impacto sobre la Privacidad (PIA) y pruebas de seguridad previas al lanzamiento.
    9. **Alta Dirección**
       - Impacto: Crítico | Probabilidad: Alta | Risk Score: 20
       - *Ejemplo:* Decisiones estratégicas sin considerar riesgos regulatorios, afectando la continuidad del negocio.
       - *Acciones:* Integrar la gestión de riesgos en la estrategia corporativa y definir claramente la tolerancia al riesgo.
    
    ## Revisión y Actualización
    - **Periodicidad:** Revisión anual o tras cambios significativos en el entorno regulatorio o interno.
    - **Seguimiento:** Establecer reuniones trimestrales de revisión para ajustar niveles de riesgo y evaluar la efectividad de las medidas.
    
    ## Conclusión
    Esta matriz robusta integra los parámetros básicos (impacto y probabilidad) y dimensiones adicionales (efectividad de controles, coste potencial, responsables y plazos) para proporcionar una evaluación integral y alineada con estándares internacionales (ISO 31000, COSO). Permite identificar, cuantificar y priorizar riesgos, facilitando la asignación de recursos y la implementación de acciones correctivas proactivas.
    """
    
    return risk_matrix

#@risk_assessment_agent.tool
#@limit_tool_calls(max_calls=2)
#async def cross_reference_sector_cases(ctx: RunContext[RiskAssessmentDeps], sector: str) -> str:
#    """
#    Busca casos similares o precedentes relacionados con empresas del mismo sector.
#    Limitado a 2 llamadas por sesión.
#    """
#    try:
#        query = f"casos sanciones compliance {sector}"
#        query_embedding = await get_embedding(query, ctx.deps.openai_client)
        
#        if not any(query_embedding):
#            logger.warning("Received a zero embedding vector. Skipping case search.")
#            return f"No se encontraron casos de referencia para el sector: {sector}."
        
#        result = ctx.deps.supabase.rpc(
#            'match_site_pages',
#            {
#                'query_embedding': query_embedding,
#                'match_count': 2,  # Limitamos a 2 casos relevantes
#                'filter': {}
#            }
#        ).execute()
        
#        if not result.data:
#            # Proporcionamos información básica de casos comunes en el sector
#            return f"""
#            # Casos comunes en el sector {sector}
            
#            ## Caso 1: Incumplimiento RGPD - Tratamiento excesivo de datos
#            Una empresa del sector {sector} fue sancionada por recopilar y procesar datos excesivos 
#            de clientes sin base legal adecuada ni principio de minimización.
#            - **Sanción**: Entre 10.000€ y 100.000€
#            - **Áreas afectadas**: Marketing, Sistemas, Legal
            
#            ## Caso 2: Falta de medidas de seguridad
#            Otra empresa del sector experimentó una brecha de seguridad que afectó a datos personales. 
#            La investigación reveló que no tenían implementadas medidas técnicas y organizativas adecuadas.
#            - **Sanción**: Entre 50.000€ y 300.000€
#            - **Áreas afectadas**: IT, Seguridad, Alta Dirección
            
#            Estos casos son representativos del sector y deben considerarse en su evaluación de riesgos.
#            """
        
#        formatted_cases = []
#        for doc in result.data:
#            case_text = f"""
#            # {doc['title']}
            
#            {doc['content']}
#            """
#            formatted_cases.append(case_text)
            
#        combined_text = "\n\n---\n\n".join(formatted_cases)
#        total_tokens = count_tokens(combined_text, llm)
        
#        if total_tokens > MAX_TOTAL_TOKENS:
#            truncated_cases = []
#            accumulated_tokens = 0
#            for case in formatted_cases:
#                case_tokens = count_tokens(case, llm)
#                if accumulated_tokens + case_tokens > MAX_TOTAL_TOKENS:
#                    remaining_tokens = MAX_TOTAL_TOKENS - accumulated_tokens
#                    truncated_case = truncate_text(case, remaining_tokens, llm)
#                    truncated_cases.append(truncated_case)
#                    break
#                else:
#                    truncated_cases.append(case)
#                    accumulated_tokens += case_tokens
#            combined_text = "\n\n---\n\n".join(truncated_cases)
        
#        return combined_text
    
#    except Exception as e:
#        logger.error(f"Error retrieving sector cases: {e}")
#        return f"Error al buscar casos de referencia para el sector: {sector}. Considera una evaluación de riesgos basada en patrones comunes del sector."


