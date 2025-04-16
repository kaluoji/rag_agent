from fastapi import Depends
from typing import List, Dict, Any, Optional
import logging
import os
import uuid
import json
import asyncio
import mammoth
from datetime import datetime
from uuid import UUID

from app.core.config import settings
from app.models.schemas import ReportResponse, AnnotationBase, Annotation
from app.core.websocket import ConnectionManager
from app.services.agent_service import AgentService, get_agent_service

# Importaciones de tu sistema multi-agente
from agents.report_agent import process_report_query, ReportDeps
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ReportService:
    """
    Servicio para la generación y gestión de reportes.
    """
    
    def __init__(self):
        """
        Inicializa el servicio de reportes.
        """
        # Asegurar que el directorio de reportes existe
        os.makedirs(settings.REPORTS_DIR, exist_ok=True)
        
        # Directorio para metadatos de reportes
        self.metadata_dir = os.path.join(settings.REPORTS_DIR, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Inicializar cliente de OpenAI
        self.openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        
        # Preparar dependencias para el generador de reportes
        self.report_deps = ReportDeps(
            output_folder=settings.REPORTS_DIR,
            openai_client=self.openai_client
        )
        
        logger.info("Servicio de reportes inicializado correctamente")
    
    async def generate_report(
        self, 
        query: str, 
        report_id: UUID, 
        format: str = "docx",
        agent_service: Optional[AgentService] = None,
        connection_manager: Optional[ConnectionManager] = None
    ):
        """
        Genera un reporte basado en una consulta.
        Esta función está diseñada para ejecutarse en segundo plano.
        
        Args:
            query: La consulta del usuario
            report_id: ID único del reporte
            format: Formato del reporte (docx, pdf)
            agent_service: Servicio de agente para obtener información normativa
            connection_manager: Gestor de conexiones WebSocket para enviar actualizaciones
        """
        try:
            logger.info(f"Iniciando generación de reporte {report_id} para consulta: {query[:100]}...")
            
            # Preparar la información inicial del reporte
            report_info = ReportResponse(
                report_id=report_id,
                query=query,
                status="generating",
                timestamp=datetime.now(),
                message="Generando reporte..."
            )
            
            # Guardar información del reporte
            await self._save_report_info(report_info)
            
            # Enviar actualización de estado
            if connection_manager:
                await connection_manager.broadcast_to_group(
                    f"report_{report_id}",
                    {"status": "generating", "message": "Generando reporte..."}
                )
            
            # Si no tenemos el servicio de agente, lo obtenemos
            if agent_service is None:
                agent_service = get_agent_service()
            
            # Primero obtenemos la información normativa a través del agente
            response, metadata = await agent_service.process_query(query)
            
            # Enviar actualización de progreso
            if connection_manager:
                await connection_manager.broadcast_to_group(
                    f"report_{report_id}",
                    {"status": "generating", "message": "Información normativa obtenida, generando documento..."}
                )
            
            # Generar el reporte usando tu agente de reportes existente
            report_result = await process_report_query(
                query=query,
                analysis_data=response,
                deps=self.report_deps,
                format_type="word"  # Siempre usamos Word como formato base
            )
            
            # Actualizar la información del reporte
            report_info.status = "ready"
            report_info.report_path = report_result.file_path
            report_info.message = report_result.message
            
            # Guardar la información actualizada
            await self._save_report_info(report_info)
            
            # Generar HTML para previsualización
            html_content = await self._generate_html_preview(report_result.file_path)
            
            # Guardar contenido HTML para previsualización
            html_file_path = os.path.join(settings.REPORTS_DIR, f"{report_id}.html")
            with open(html_file_path, "w", encoding="utf-8") as html_file:
                html_file.write(html_content)
            
            # Enviar actualización de finalización con ruta al reporte
            if connection_manager:
                await connection_manager.broadcast_to_group(
                    f"report_{report_id}",
                    {
                        "status": "ready", 
                        "message": "Reporte generado correctamente",
                        "reportPath": report_result.file_path,
                        "reportHtml": html_content
                    }
                )
            
            logger.info(f"Reporte {report_id} generado correctamente: {report_result.file_path}")
            
        except Exception as e:
            logger.error(f"Error al generar reporte {report_id}: {str(e)}")
            
            # Actualizar información del reporte con error
            if report_info:
                report_info.status = "error"
                report_info.message = f"Error al generar el reporte: {str(e)}"
                await self._save_report_info(report_info)
            
            # Enviar notificación de error
            if connection_manager:
                await connection_manager.broadcast_to_group(
                    f"report_{report_id}",
                    {"status": "error", "message": f"Error al generar el reporte: {str(e)}"}
                )
    
    async def _save_report_info(self, report_info: ReportResponse):
        """
        Guarda la información del reporte en un archivo JSON.
        
        Args:
            report_info: Información del reporte a guardar
        """
        try:
            file_path = os.path.join(self.metadata_dir, f"{report_info.report_id}.json")
            
            # Convertir a diccionario compatible con JSON
            report_dict = report_info.dict()
            # Convertir UUID a string
            report_dict["report_id"] = str(report_dict["report_id"])
            # Convertir datetime a string
            report_dict["timestamp"] = report_dict["timestamp"].isoformat()
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error al guardar información del reporte: {str(e)}")
    
    async def get_report_info(self, report_id: UUID) -> Optional[ReportResponse]:
        """
        Obtiene la información de un reporte.
        
        Args:
            report_id: ID único del reporte
            
        Returns:
            Optional[ReportResponse]: Información del reporte o None si no existe
        """
        try:
            file_path = os.path.join(self.metadata_dir, f"{report_id}.json")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)
            
            # Convertir string a UUID
            report_dict["report_id"] = uuid.UUID(report_dict["report_id"])
            # Convertir string a datetime
            report_dict["timestamp"] = datetime.fromisoformat(report_dict["timestamp"])
            
            return ReportResponse(**report_dict)
            
        except Exception as e:
            logger.error(f"Error al obtener información del reporte {report_id}: {str(e)}")
            return None
    
    async def _generate_html_preview(self, docx_path: str) -> str:
        """
        Genera una previsualización HTML de un documento Word.
        
        Args:
            docx_path: Ruta al archivo DOCX
            
        Returns:
            str: Contenido HTML del documento
        """
        try:
            with open(docx_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value
                
                # Envolver el HTML con estilos básicos
                wrapped_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Vista previa del reporte</title>
                    <style>
                        body {{
                            font-family: 'Times New Roman', Times, serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }}
                        h1, h2, h3, h4, h5, h6 {{
                            margin-top: 1.2em;
                            margin-bottom: 0.6em;
                        }}
                        p {{
                            margin-bottom: 1em;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin-bottom: 1em;
                        }}
                        table, th, td {{
                            border: 1px solid #ddd;
                        }}
                        th, td {{
                            padding: 8px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #f2f2f2;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                    </style>
                </head>
                <body class="document-preview">
                    {html}
                </body>
                </html>
                """
                
                return wrapped_html
                
        except Exception as e:
            logger.error(f"Error al generar previsualización HTML: {str(e)}")
            return f"<p>Error al generar la previsualización: {str(e)}</p>"
    
    async def get_report_html(self, report_id: UUID) -> Optional[str]:
        """
        Obtiene la previsualización HTML de un reporte.
        
        Args:
            report_id: ID único del reporte
            
        Returns:
            Optional[str]: Contenido HTML del reporte o None si no existe
        """
        try:
            # Primero verificamos si ya existe el archivo HTML
            html_path = os.path.join(settings.REPORTS_DIR, f"{report_id}.html")
            
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            # Si no existe, obtenemos la información del reporte
            report_info = await self.get_report_info(report_id)
            
            if not report_info or not report_info.report_path or not os.path.exists(report_info.report_path):
                return None
            
            # Generar la previsualización HTML
            html_content = await self._generate_html_preview(report_info.report_path)
            
            # Guardar el contenido HTML para futuras solicitudes
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error al obtener previsualización HTML del reporte {report_id}: {str(e)}")
            return None
    
    async def save_annotations(self, report_id: UUID, annotations: List[AnnotationBase]) -> List[Annotation]:
        """
        Guarda anotaciones asociadas a un reporte.
        
        Args:
            report_id: ID único del reporte
            annotations: Lista de anotaciones a guardar
            
        Returns:
            List[Annotation]: Lista de anotaciones guardadas con sus IDs
        """
        try:
            # Verificar que el reporte existe
            report_info = await self.get_report_info(report_id)
            
            if not report_info:
                raise ValueError(f"Reporte con ID {report_id} no encontrado")
            
            # Archivo de anotaciones
            annotations_file = os.path.join(self.metadata_dir, f"{report_id}_annotations.json")
            
            # Cargar anotaciones existentes si existen
            existing_annotations = []
            if os.path.exists(annotations_file):
                with open(annotations_file, "r", encoding="utf-8") as f:
                    existing_annotations_data = json.load(f)
                    
                    # Convertir a objetos Annotation
                    for ann_data in existing_annotations_data:
                        ann_data["id"] = uuid.UUID(ann_data["id"])
                        ann_data["report_id"] = uuid.UUID(ann_data["report_id"])
                        ann_data["timestamp"] = datetime.fromisoformat(ann_data["timestamp"])
                        existing_annotations.append(Annotation(**ann_data))
            
            # Crear nuevas anotaciones completas
            saved_annotations = []
            for annotation in annotations:
                new_annotation = Annotation(
                    id=uuid.uuid4(),
                    report_id=report_id,
                    selected_text=annotation.selected_text,
                    annotation_text=annotation.annotation_text,
                    timestamp=datetime.now()
                )
                saved_annotations.append(new_annotation)
            
            # Combinar con las existentes
            all_annotations = existing_annotations + saved_annotations
            
            # Preparar para guardar en JSON
            annotations_to_save = []
            for ann in all_annotations:
                ann_dict = ann.dict()
                ann_dict["id"] = str(ann_dict["id"])
                ann_dict["report_id"] = str(ann_dict["report_id"])
                ann_dict["timestamp"] = ann_dict["timestamp"].isoformat()
                annotations_to_save.append(ann_dict)
            
            # Guardar todas las anotaciones
            with open(annotations_file, "w", encoding="utf-8") as f:
                json.dump(annotations_to_save, f, ensure_ascii=False, indent=2)
            
            return saved_annotations
            
        except Exception as e:
            logger.error(f"Error al guardar anotaciones para el reporte {report_id}: {str(e)}")
            raise

# Singleton del servicio
_report_service = None

def get_report_service() -> ReportService:
    """
    Devuelve una instancia del servicio de reportes (singleton).
    """
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service