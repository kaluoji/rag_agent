from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
import uuid
from datetime import datetime
from app.models.schemas import QueryRequest, QueryResponse
from app.services.agent_service import AgentService, get_agent_service
from app.models.schemas import QueryRequest, QueryResponse, QueryWithDocumentsRequest

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

@router.post("/with-documents", response_model=QueryResponse)
async def process_query_with_documents(
    query_request: QueryWithDocumentsRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Procesa una consulta normativa con documentos adjuntos para análisis GAP.
    """
    try:
        logger.info(f"Procesando consulta con documentos: {query_request.query[:100]}...")
        
        # Generar un ID único para la consulta
        query_id = uuid.uuid4()
        
        # Procesar documentos si están presentes
        processed_documents = []
        if query_request.documents:
            for doc in query_request.documents:
                try:
                    # Decodificar el contenido base64
                    import base64
                    decoded_content = base64.b64decode(doc.content).decode('utf-8', errors='ignore')
                    
                    processed_documents.append({
                        "name": doc.name,
                        "type": doc.type,
                        "content": decoded_content,
                        "size": doc.size
                    })
                    
                    logger.info(f"Documento procesado: {doc.name} ({doc.size} bytes)")
                    
                except Exception as e:
                    logger.warning(f"Error procesando documento {doc.name}: {e}")
                    # Continuar con otros documentos
                    continue
        
        # Enriquecer la consulta con el contenido de los documentos
        enriched_query = query_request.query
        if processed_documents:
            # Agregar el contenido del primer documento a la consulta
            doc = processed_documents[0]
            enriched_query = f"""Realiza un análisis GAP del siguiente documento de política interna.

DOCUMENTO ANALIZAR: {doc['name']}

CONTENIDO DEL DOCUMENTO:
{doc['content'][:20000]}  # Limitamos a 20k caracteres para evitar problemas con tokens

CONSULTA ESPECÍFICA: {query_request.query}

Por favor, utiliza la herramienta perform_gap_analysis para comparar este documento con la normativa aplicable."""
        
        # Procesar la consulta con el sistema multi-agente
        response, metadata = await agent_service.process_query(enriched_query)
        
        # Añadir información sobre documentos procesados a los metadatos
        if processed_documents:
            metadata["processed_documents"] = [
                {"name": doc["name"], "size": doc["size"]} 
                for doc in processed_documents
            ]
        
        # Construir y devolver la respuesta
        query_response = QueryResponse(
            response=response,
            query=query_request.query,  # Devolver la consulta original, no la enriquecida
            query_id=query_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        return query_response
    
    except Exception as e:
        logger.error(f"Error al procesar consulta con documentos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la consulta con documentos: {str(e)}"
        )