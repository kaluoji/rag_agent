from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path
from fastapi.responses import FileResponse, HTMLResponse
from typing import List, Dict, Any
import logging
import uuid
import os
from datetime import datetime
from app.models.schemas import ReportRequest, ReportResponse, AnnotationsRequest, Annotation
from app.services.report_service import ReportService, get_report_service
from app.services.agent_service import AgentService, get_agent_service
from app.core.config import settings
from app.core.websocket import ConnectionManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Obtener el gestor de conexiones WebSocket desde main.py
connection_manager = None

@router.on_event("startup")
def startup_event():
    global connection_manager
    from app.main import connection_manager as cm
    connection_manager = cm

@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Inicia la generación de un reporte en segundo plano y devuelve un identificador
    para seguir el progreso.
    """
    try:
        # Generar un ID único para el reporte
        report_id = uuid.uuid4()
        
        # Crear la respuesta inicial
        response = ReportResponse(
            report_id=report_id,
            query=report_request.query,
            status="generating",
            timestamp=datetime.now(),
            estimated_time=60,  # Estimar 60 segundos por defecto
            message="Generación de reporte iniciada"
        )
        
        # Iniciar la generación del reporte en segundo plano
        background_tasks.add_task(
            report_service.generate_report,
            query=report_request.query,
            report_id=report_id,
            format=report_request.format,
            agent_service=agent_service,
            connection_manager=connection_manager
        )
        
        logger.info(f"Generación de reporte iniciada. ID: {report_id}")
        return response
    
    except Exception as e:
        logger.error(f"Error al iniciar generación de reporte: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al iniciar la generación del reporte: {str(e)}"
        )

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report_status(
    report_id: uuid.UUID = Path(...),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Obtiene el estado actual de un reporte en proceso o completado.
    """
    try:
        report_info = await report_service.get_report_info(report_id)
        if not report_info:
            raise HTTPException(
                status_code=404,
                detail=f"Reporte con ID {report_id} no encontrado"
            )
        
        return report_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener estado del reporte: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estado del reporte: {str(e)}"
        )

@router.get("/preview/{report_id}", response_class=HTMLResponse)
async def preview_report(
    report_id: uuid.UUID = Path(...),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Obtiene una vista previa HTML del reporte.
    """
    try:
        html_content = await report_service.get_report_html(report_id)
        if not html_content:
            raise HTTPException(
                status_code=404,
                detail=f"Vista previa del reporte con ID {report_id} no disponible"
            )
        
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener vista previa del reporte: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener vista previa del reporte: {str(e)}"
        )

@router.get("/{report_id}/download")
async def download_report(
    report_id: uuid.UUID = Path(...),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Descarga el archivo del reporte en su formato original.
    """
    try:
        report_info = await report_service.get_report_info(report_id)
        if not report_info or not report_info.report_path or report_info.status != "ready":
            raise HTTPException(
                status_code=404,
                detail=f"Reporte con ID {report_id} no disponible para descarga"
            )
        
        file_path = report_info.report_path
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Archivo del reporte no encontrado"
            )
        
        # Obtener el nombre del archivo
        filename = os.path.basename(file_path)
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al descargar el reporte: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al descargar el reporte: {str(e)}"
        )

@router.post("/{report_id}/annotations", response_model=List[Annotation])
async def save_annotations(
    annotations_request: AnnotationsRequest,
    report_id: uuid.UUID = Path(...),
    report_service: ReportService = Depends(get_report_service)
):
    """
    Guarda anotaciones asociadas a un reporte.
    """
    try:
        # Verificar que el reporte existe
        report_info = await report_service.get_report_info(report_id)
        if not report_info:
            raise HTTPException(
                status_code=404,
                detail=f"Reporte con ID {report_id} no encontrado"
            )
        
        # Guardar las anotaciones
        saved_annotations = await report_service.save_annotations(
            report_id=report_id,
            annotations=annotations_request.annotations
        )
        
        return saved_annotations
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al guardar anotaciones: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al guardar anotaciones: {str(e)}"
        )