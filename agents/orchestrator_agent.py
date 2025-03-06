from __future__ import annotations as _annotations

import logging
import json
from enum import Enum
from typing import List, Dict, Optional, Union, Any, Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from supabase import Client

from config.config import settings
from agents.ai_expert_v1 import ai_expert, AIDeps, debug_run_agent  # Importamos la función debug_run_agent
from agents.risk_assessment_agent import risk_assessment_agent, RiskAssessmentDeps
from agents.report_agent import report_agent, ReportDeps

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización del modelo
llm = settings.llm_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

class AgentType(str, Enum):
    """Tipos de agentes disponibles en el sistema."""
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    REPORT = "report"
    

class OrchestratorDeps(BaseModel):
    """Dependencias necesarias para el agente orquestador."""
    supabase: Client
    openai_client: AsyncOpenAI

    class Config:
        arbitrary_types_allowed = True

class AgentInfo(BaseModel):
    """Información sobre un agente detectado en la consulta."""
    agent_type: AgentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    query_parameters: Dict[str, Any] = {}

class OrchestrationResult(BaseModel):
    """Resultado de la orquestación."""
    agent_used: AgentType
    response: Any
    additional_info: Optional[Dict[str, Any]] = None

system_prompt = """
Eres un agente orquestador encargado de coordinar múltiples agentes especializados en medios de pago y en las reglas de Visa y Mastercard. Tu objetivo principal es:

Analizar la consulta del usuario para identificar su intención principal.
Determinar cuál de los agentes disponibles es el más adecuado para atenderla.
Proporcionar tu nivel de confianza (entre 0.0 y 1.0) en dicha selección.
Identificar parámetros o información adicional para el agente elegido (si aplica).
Agente disponible:

Agente de Compliance (COMPLIANCE)
Experto en normativas de cumplimiento (compliance), especializado en reglas de Visa y Mastercard.
Ideal para:
Consultas sobre reglas de Visa/Mastercard.
Explicaciones de obligaciones regulatorias.
Dudas específicas sobre cumplimiento normativo.
Generación de informes relacionados con normas y regulaciones.
Traductor de idioma en caso de que la consulta esté en un idioma diferente al español.

Agente de Evaluación de Riesgos (RISK_ASSESSMENT) – Especialista en analizar riesgos por sector. Ideal para:
Identificación de áreas de riesgo en una empresa
Evaluación de impacto y probabilidad de riesgos regulatorios
Análisis de riesgos específicos del sector
Recomendaciones para mitigar riesgos de compliance
Realización de GAP analysis con respecto a la información de la BBDD

Instrucciones Clave:
Enfócate en la intención principal de la consulta.
Si la consulta toca varias áreas, prioriza la necesidad más relevante o el objetivo final del usuario.
Si la consulta requiere un GAP analysis respecto a la normativa de BBDD, selecciona siempre al Agente COMPLIANCE.
Para consultas generales sobre reglas de Visa o Mastercard, prioriza igualmente al Agente COMPLIANCE.
Para consultas que soliciten explícitamente evaluación de riesgos por sector o análisis de áreas impactadas, prioriza el agente RISK_ASSESSMENT.

Importante:
No respondas directamente la consulta del usuario.
Tu única función es redirigir la consulta al agente especializado adecuado.
"""


orchestrator_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=OrchestratorDeps,
    result_type=AgentInfo
)

async def route_to_agent(agent_info: AgentInfo, deps: OrchestratorDeps, query: str) -> OrchestrationResult:
    """
    Enruta la consulta al agente adecuado según el análisis del orquestador.
    """
    # Verificación para consultas en inglés
    english_keywords = ["what", "how", "when", "where", "who", "why", "which", "is", "are", "can", "could", "do", "does"]
    is_english = any(keyword.lower() in query.lower().split() for keyword in english_keywords)
    
    # Forzar redirección a COMPLIANCE para todas las consultas en inglés
    if is_english and agent_info.agent_type != AgentType.COMPLIANCE:
        logger.warning(f"ORQUESTADOR: Consulta en inglés detectada, redirigiendo de {agent_info.agent_type} a COMPLIANCE")
        agent_info.agent_type = AgentType.COMPLIANCE

    logger.info(f"Enrutando consulta al agente: {agent_info.agent_type} (confianza: {agent_info.confidence})")
    
    # Crear las dependencias para el agente de compliance
    ai_deps = AIDeps(
        supabase=deps.supabase,
        openai_client=deps.openai_client
    )
    
    if agent_info.agent_type == AgentType.COMPLIANCE:
        # Redirigir al agente de compliance usando debug_run_agent
        response = await debug_run_agent(query, deps=ai_deps)
        
        return OrchestrationResult(
            agent_used=AgentType.COMPLIANCE,
            response=response.data,
            additional_info={"usage": response.usage()}
        )
    
    elif agent_info.agent_type == AgentType.RISK_ASSESSMENT:
        # Redirigir al agente de evaluación de riesgos
        risk_deps = RiskAssessmentDeps(
            supabase=deps.supabase,
            openai_client=deps.openai_client
        )
        response = await risk_assessment_agent.run(
            query,
            deps=risk_deps
        )
        return OrchestrationResult(
            agent_used=AgentType.RISK_ASSESSMENT,
            response=response.data,
            additional_info={"usage": response.usage()}
        )
    
    elif agent_info.agent_type == AgentType.REPORT:
        # Redirigir al agente de informes
        report_deps = ReportDeps(
            openai_client=deps.openai_client
        )
        response = await report_agent.run(
            query,
            deps=report_deps
        )
        return OrchestrationResult(
            agent_used=AgentType.REPORT,
            response=response.data,
            additional_info={"report_type": agent_info.query_parameters.get("report_type", "word")}
        )
    
    else:
        # Si no se reconoce el tipo de agente, usamos el agente de compliance por defecto
        logger.warning(f"Tipo de agente desconocido: {agent_info.agent_type}. Usando agente de compliance por defecto.")
        response = await debug_run_agent(query, deps=ai_deps)
        
        return OrchestrationResult(
            agent_used=AgentType.COMPLIANCE,
            response=response.data,
            additional_info={"usage": response.usage()}
        )

async def process_query(query: str, deps: OrchestratorDeps) -> OrchestrationResult:
    """
    Procesa una consulta de usuario a través del sistema multi-agente.
    
    Args:
        query: Consulta del usuario
        deps: Dependencias necesarias para los agentes
    
    Returns:
        OrchestrationResult: Resultado de la ejecución del agente seleccionado
    """
    # Log de la consulta original recibida
    logger.info("=" * 50)
    logger.info(f"ORQUESTADOR: Consulta recibida: {query[:100]}..." if len(query) > 100 else query)
    logger.info("=" * 50)
    
    # El orquestador analiza la consulta y determina qué agente debe manejarla
    logger.info("ORQUESTADOR: Analizando consulta con el agente orquestador...")
    orchestration_result = await orchestrator_agent.run(
        query,
        deps=deps
    )
    
    agent_info = orchestration_result.data
    logger.info(f"ORQUESTADOR: Decisión del orquestador - Agente seleccionado: {agent_info.agent_type} con confianza: {agent_info.confidence}")
    
    # Log de parámetros adicionales identificados por el orquestador
    if agent_info.query_parameters:
        logger.info(f"ORQUESTADOR: Parámetros detectados: {agent_info.query_parameters}")
    
    # Redirigimos la consulta al agente especializado
    logger.info(f"ORQUESTADOR: Redirigiendo consulta al agente {agent_info.agent_type}...")
    return await route_to_agent(agent_info, deps, query)