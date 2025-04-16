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

from app.core.config import settings
from agents.ai_expert_v1 import ai_expert, AIDeps, debug_run_agent
from agents.understanding_query import process_query as process_query_understanding, QueryUnderstandingDeps, QueryInfo
from agents.report_agent import report_agent, ReportDeps, process_report_query
#from agents.risk_assessment_agent import risk_assessment_agent, RiskAssessmentDeps
#from agents.normative_report_agent import normative_report_agent, ReportDeps as NormativeReportDeps, process_report_query

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización del modelo
llm = settings.llm_model
model = OpenAIModel(
    llm, 
    api_key=settings.openai_api_key
)

class AgentType(str, Enum):
    """Tipos de agentes disponibles en el sistema."""
    COMPLIANCE = "compliance"
    QUERY_UNDERSTANDING = "query_understanding"
    REPORT = "report"
    #RISK_ASSESSMENT = "risk_assessment"
    #NORMATIVE_REPORT = "normative_report"
    

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

class OrchestratorPlan(BaseModel):
    """Plan de ejecución generado por el orquestador."""
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    requires_query_understanding: bool = False
    requires_complex_handling: bool = False
    primary_agent: AgentType
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class OrchestrationResult(BaseModel):
    """Resultado de la orquestación."""
    agent_used: AgentType
    response: Any
    query_info: Optional[QueryInfo] = None
    additional_info: Optional[Dict[str, Any]] = None

system_prompt = """
Eres un agente orquestador encargado de coordinar múltiples agentes especializados en medios de pago y en las reglas de Visa y Mastercard. Tu objetivo principal es:

1. Determinar qué agentes deben involucrarse en la respuesta a una consulta
2. Planificar el flujo de trabajo necesario para responder a la consulta
3. Especificar los parámetros o información adicional para cada agente

Agentes disponibles:

1. QUERY_UNDERSTANDING
   - Especializado en analizar y procesar consultas de usuario
   - Expande consultas con términos implícitos
   - Descompone consultas complejas en sub-consultas
   - Identifica entidades e intenciones en la consulta
   - Detecta idioma y evalúa complejidad

2. COMPLIANCE
   - Experto en normativas de cumplimiento, especializado en reglas de Visa y Mastercard
   - Ideal para consultas sobre regulaciones, obligaciones y procesos normativos
   - Puede generar informes relacionados con normas de cumplimiento

3. REPORT
   - Especializado en generar reportes normativos formales en formato Word
   - Útil cuando el usuario solicita explícitamente un informe o documento 
   - Necesita trabajar con COMPLIANCE para obtener el contenido regulatorio
   - Genera documentos estructurados con secciones estándar (introducción, alcance, análisis, etc.)

Para cada consulta, debes determinar:
1. Si se requiere un análisis previo con QUERY_UNDERSTANDING
2. Si la consulta parece compleja y podría beneficiarse de la descomposición
3. Si el usuario está solicitando un reporte formal (usar REPORT) o solo información (usar COMPLIANCE)
4. Qué agente principal debe responder la consulta

Tu respuesta debe ser un plan de orquestación que especifique la secuencia de agentes a utilizar y cualquier información relevante para su ejecución.

Importante:
No respondas directamente la consulta del usuario.
Tu única función es planificar la coordinación entre los agentes especializados.
"""

orchestrator_agent = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=OrchestratorDeps,
    result_type=OrchestratorPlan
)

async def execute_orchestration_plan(plan: OrchestratorPlan, query: str, deps: OrchestratorDeps) -> OrchestrationResult:
    """
    Ejecuta un plan de orquestación generado por el orquestador.
    
    Args:
        plan: Plan de orquestación a ejecutar
        query: Consulta original del usuario
        deps: Dependencias necesarias para los agentes
    
    Returns:
        OrchestrationResult: Resultado de la ejecución del plan
    """
    logger.info(f"Ejecutando plan de orquestación - Agente principal: {plan.primary_agent}")
    query_info = None
    effective_query = query
    
    # Preparar dependencias para los diferentes agentes
    ai_deps = AIDeps(
        supabase=deps.supabase,
        openai_client=deps.openai_client
    )
    
    query_understanding_deps = QueryUnderstandingDeps(
        supabase=deps.supabase,
        openai_client=deps.openai_client
    )

    report_deps = ReportDeps(
        output_folder="output/reports",
        openai_client=deps.openai_client
    )
    
    # Paso 1: Procesar con Query Understanding si el plan lo requiere
    if plan.requires_query_understanding:
        logger.info("Ejecutando paso de comprensión de consulta (Query Understanding)")
        query_info = await process_query_understanding(query, deps=query_understanding_deps)
        
        # Si tenemos una consulta expandida, la usamos
        if query_info.expanded_query:
            effective_query = query_info.expanded_query
            logger.info(f"Usando consulta expandida: {effective_query[:100]}..." if len(effective_query) > 100 else effective_query)
    
    # Paso 2: Si la consulta es compleja y tenemos información de Query Understanding, 
    # procesamos cada sub-consulta
    if plan.requires_complex_handling and query_info and query_info.decomposed_queries:
        logger.info(f"Procesando consulta compleja con {len(query_info.decomposed_queries)} sub-consultas")
        
        # Procesar cada sub-consulta
        sub_responses = []
        for i, sub_query in enumerate(query_info.decomposed_queries):
            logger.info(f"Procesando sub-consulta {i+1}: {sub_query[:100]}..." if len(sub_query) > 100 else sub_query)
            
            if plan.primary_agent == AgentType.COMPLIANCE:
                sub_response = await debug_run_agent(sub_query, deps=ai_deps)
                sub_responses.append(sub_response.data)
        
        # Sintetizar los resultados de las sub-consultas
        synthesis_prompt = f"Sintetiza de manera coherente las siguientes respuestas a sub-consultas sobre: {query}\n\n"
        for i, sub_response in enumerate(sub_responses):
            synthesis_prompt += f"Respuesta {i+1} (a la sub-consulta: {query_info.decomposed_queries[i]}):\n{sub_response}\n\n"
        
        # Usar el modelo para sintetizar una respuesta coherente
        completion = await deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "Eres un experto en normativas de VISA y Mastercard encargado de sintetizar múltiples respuestas en una única respuesta coherente y completa. Conserva toda la información relevante, elimina redundancias y organiza el contenido de manera lógica."},
                {"role": "user", "content": synthesis_prompt}
            ]
        )
        
        synthesized_response = completion.choices[0].message.content
        
        # Devolver la respuesta sintetizada
        return OrchestrationResult(
            agent_used=plan.primary_agent,
            response=synthesized_response,
            query_info=query_info,
            additional_info={"processed_sub_queries": len(sub_responses)}
        )
    
    # Paso 3: Procesamiento estándar con el agente principal
    logger.info(f"Procesando con el agente principal: {plan.primary_agent}")
    
    if plan.primary_agent == AgentType.COMPLIANCE:
        response = await debug_run_agent(effective_query, deps=ai_deps, query_info=query_info)
        
        return OrchestrationResult(
            agent_used=plan.primary_agent,
            response=response.data,
            query_info=query_info,
            additional_info={"usage": response.usage()}
        )

    elif plan.primary_agent == AgentType.REPORT:
        # Primero obtenemos el análisis de compliance
        logger.info("Generando contenido con el agente de compliance para el reporte")
        compliance_response = await debug_run_agent(effective_query, deps=ai_deps, query_info=query_info)
        
        # Luego generamos el reporte con ese contenido
        logger.info("Generando reporte en formato Word con el contenido de compliance")
        report_result = await process_report_query(
            query=effective_query,
            analysis_data=compliance_response.data,
            deps=report_deps,
            format_type="word"  # Forzamos el formato Word siempre
        )
        
        # Preparar respuesta apropiada
        report_message = f"""He generado un informe normativo basado en tu consulta.

Título: {report_result.file_path.split('/')[-1].replace('_', ' ').replace('.docx', '')}
Estado: {report_result.message}
Ubicación: {report_result.file_path}

El informe incluye un análisis detallado de las normativas VISA y Mastercard relevantes, conclusiones y recomendaciones.
"""
        
        return OrchestrationResult(
            agent_used=plan.primary_agent,
            response=report_message,
            query_info=query_info,
            additional_info={"report_path": report_result.file_path}
        )

    else:
        # Si no se reconoce el tipo de agente, usamos el agente de compliance por defecto
        logger.warning(f"Tipo de agente principal no implementado: {plan.primary_agent}. Usando agente de compliance por defecto.")
        response = await debug_run_agent(effective_query, deps=ai_deps)
        
        return OrchestrationResult(
            agent_used=AgentType.COMPLIANCE,
            response=response.data,
            query_info=query_info,
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
    
    # El orquestador analiza la consulta y genera un plan de ejecución
    logger.info("ORQUESTADOR: Generando plan de ejecución...")
    orchestration_result = await orchestrator_agent.run(
        query,
        deps=deps
    )
    
    plan = orchestration_result.data
    logger.info(f"ORQUESTADOR: Plan generado - Agente principal: {plan.primary_agent}")
    logger.info(f"ORQUESTADOR: ¿Requiere Query Understanding? {plan.requires_query_understanding}")
    logger.info(f"ORQUESTADOR: ¿Requiere manejo de consulta compleja? {plan.requires_complex_handling}")
    
    if plan.steps:
        for i, step in enumerate(plan.steps):
            logger.info(f"ORQUESTADOR: Paso {i+1}: {step}")
    
    # Ejecutar el plan de orquestación
    logger.info("ORQUESTADOR: Ejecutando plan de orquestación...")
    result = await execute_orchestration_plan(plan, query, deps)
    
    return result