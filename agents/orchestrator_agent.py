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
    UNKNOWN = "unknown"

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
Eres un agente orquestador altamente eficiente que coordina entre múltiples agentes especializados en el área de compliance y protección de datos.

Tu función principal es analizar la consulta del usuario, determinar qué agente especializado es el más adecuado para manejarla, y dirigirla al agente correspondiente.

Los agentes disponibles son:

1. **Agente de Compliance (COMPLIANCE)** - Experto en regulación de protección de datos y normativas de privacidad. Ideal para:
   - Consultas sobre leyes específicas (RGPD, LOPDGDD, etc.)
   - Explicaciones de obligaciones regulatorias
   - Respuestas a dudas específicas sobre cumplimiento normativo
   - Generación de informes sobre normativas

2. **Agente de Evaluación de Riesgos (RISK_ASSESSMENT)** - Especialista en analizar riesgos por sector. Ideal para:
   - Identificación de áreas de riesgo en una empresa
   - Evaluación de impacto y probabilidad de riesgos regulatorios
   - Análisis de riesgos específicos del sector
   - Recomendaciones para mitigar riesgos de compliance

3. **Agente de Informes (REPORT)** - Diseñado para generar informes estructurados. Ideal para:
   - Generación de informes en formato Word o PowerPoint
   - Estructuración formal de hallazgos y recomendaciones

**IMPORTANTE**:
- Debes determinar el agente más apropiado basándote en la intención principal de la consulta.
- Si la consulta toca múltiples áreas, enfócate en la intención principal o el objetivo final del usuario.
- Para consultas que soliciten explícitamente evaluación de riesgos por sector o análisis de áreas impactadas, prioriza el agente RISK_ASSESSMENT.
- Para consultas generales sobre normativas o dudas de compliance, prioriza el agente COMPLIANCE.
- Para solicitudes explícitas de generación de informes, prioriza el agente REPORT.

**Tu tarea es:**
1. Analizar la consulta para identificar la intención principal
2. Determinar qué agente especializado es el más apropiado
3. Proporcionar el nivel de confianza en tu decisión (de 0.0 a 1.0)
4. Identificar parámetros específicos para el agente seleccionado (si aplica)

No respondas directamente a la consulta. Tu única función es dirigir la consulta al agente especializado adecuado.
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
    # El orquestador analiza la consulta y determina qué agente debe manejarla
    orchestration_result = await orchestrator_agent.run(
        query,
        deps=deps
    )
    
    agent_info = orchestration_result.data
    logger.info(f"Agente seleccionado: {agent_info.agent_type} con confianza: {agent_info.confidence}")
    
    # Si la confianza es baja, podríamos implementar una lógica adicional
    # Por ejemplo, solicitar más información al usuario
    
    # Redirigimos la consulta al agente especializado
    return await route_to_agent(agent_info, deps, query)