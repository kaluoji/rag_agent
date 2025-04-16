from fastapi import Depends
from typing import Tuple, Dict, Any, Optional
import logging
import os

from supabase import create_client
from openai import AsyncOpenAI

from app.core.config import settings

# Importaciones de tu sistema multi-agente
from agents.orchestrator_agent import process_query as process_orchestrator_query
from agents.orchestrator_agent import OrchestratorDeps

logger = logging.getLogger(__name__)

class AgentService:
    """
    Servicio que coordina la interacción con el sistema multi-agente.
    """
    
    def __init__(self):
        """
        Inicializa el servicio de agente con las dependencias necesarias.
        """
        # Inicializar cliente de Supabase
        self.supabase = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        
        # Inicializar cliente de OpenAI
        self.openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        
        # Preparar dependencias para el orquestador
        self.orchestrator_deps = OrchestratorDeps(
            supabase=self.supabase,
            openai_client=self.openai_client
        )
        
        logger.info("Servicio de agente inicializado correctamente")
    
    async def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Procesa una consulta utilizando el sistema multi-agente.
        
        Args:
            query: La consulta del usuario
            
        Returns:
            Tuple[str, Dict[str, Any]]: La respuesta y metadatos adicionales
        """
        try:
            logger.info(f"Procesando consulta con orquestador: {query[:100]}...")
            
            # Usar el orquestador para procesar la consulta
            orchestration_result = await process_orchestrator_query(
                query=query,
                deps=self.orchestrator_deps
            )
            
            # Extraer la respuesta y los metadatos
            response = orchestration_result.response
            
            # Preparar metadatos adicionales
            metadata = {
                "agent_used": orchestration_result.agent_used,
            }
            
            # Incluir información de query_info si está disponible
            if orchestration_result.query_info:
                metadata["query_info"] = orchestration_result.query_info.dict()
            
            # Incluir información adicional si está disponible
            if orchestration_result.additional_info:
                metadata["additional_info"] = orchestration_result.additional_info
            
            logger.info(f"Consulta procesada correctamente con agente: {orchestration_result.agent_used}")
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error al procesar consulta con el sistema multi-agente: {str(e)}")
            raise

# Singleton del servicio
_agent_service = None

def get_agent_service() -> AgentService:
    """
    Devuelve una instancia del servicio de agente (singleton).
    """
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service