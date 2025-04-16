from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
import uuid
from datetime import datetime
from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_service import AgentService, get_agent_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Procesa una consulta normativa y devuelve la respuesta del agente especializado.
    """
    try:
        logger.info(f"Procesando consulta: {query_request.query[:100]}...")
        
        # Generar un ID único para la consulta
        query_id = uuid.uuid4()
        
        # Procesar la consulta con el sistema multi-agente
        response, metadata = await agent_service.process_query(query_request.query)
        
        # Construir y devolver la respuesta
        query_response = QueryResponse(
            response=response,
            query=query_request.query,
            query_id=query_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        return query_response
    
    except Exception as e:
        logger.error(f"Error al procesar consulta: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la consulta: {str(e)}"
        )