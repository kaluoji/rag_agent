from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID, uuid4

class QueryRequest(BaseModel):
    """
    Esquema para una solicitud de consulta normativa.
    """
    query: str = Field(..., description="Texto de la consulta del usuario")

class QueryResponse(BaseModel):
    """
    Esquema para la respuesta a una consulta normativa.
    """
    response: str = Field(..., description="Respuesta a la consulta")
    query: str = Field(..., description="Consulta original")
    query_id: UUID = Field(default_factory=uuid4, description="ID único de la consulta")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de la respuesta")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")

class ReportRequest(BaseModel):
    """
    Esquema para una solicitud de generación de reporte.
    """
    query: str = Field(..., description="Consulta original para generar el reporte")
    format: str = Field("docx", description="Formato del reporte (docx, pdf)")

class ReportResponse(BaseModel):
    """
    Esquema para la respuesta con información del reporte generado.
    """
    report_id: UUID = Field(default_factory=uuid4, description="ID único del reporte")
    status: str = Field("generating", description="Estado del reporte: generating, ready, error")
    query: str = Field(..., description="Consulta original")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de inicio de generación")
    estimated_time: int = Field(default=30, description="Tiempo estimado de generación en segundos")
    report_path: Optional[str] = Field(None, description="Ruta al archivo del reporte cuando esté listo")
    message: str = Field("Generación de reporte iniciada", description="Mensaje informativo")

class AnnotationBase(BaseModel):
    """
    Esquema base para una anotación sobre un reporte.
    """
    selected_text: str = Field(..., description="Texto seleccionado para anotar")
    annotation_text: str = Field(..., description="Texto de la anotación")

class AnnotationCreate(AnnotationBase):
    """
    Esquema para crear una nueva anotación.
    """
    report_id: UUID = Field(..., description="ID del reporte al que pertenece la anotación")

class Annotation(AnnotationBase):
    """
    Esquema completo de una anotación guardada.
    """
    id: UUID = Field(default_factory=uuid4, description="ID único de la anotación")
    report_id: UUID = Field(..., description="ID del reporte al que pertenece la anotación")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de creación")
    user_id: Optional[str] = Field(None, description="ID del usuario que creó la anotación")

class AnnotationsRequest(BaseModel):
    """
    Esquema para una solicitud de guardar múltiples anotaciones.
    """
    annotations: List[AnnotationBase] = Field(..., description="Lista de anotaciones a guardar")

class WebSocketMessage(BaseModel):
    """
    Esquema para mensajes enviados a través de WebSocket.
    """
    event: str = Field(..., description="Tipo de evento: update, complete, error")
    data: Dict[str, Any] = Field(default_factory=dict, description="Datos del mensaje")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp del mensaje")


class DocumentData(BaseModel):
    """Esquema para datos de documento cargado."""
    name: str = Field(..., description="Nombre del archivo")
    type: str = Field(..., description="Tipo MIME del archivo")
    content: str = Field(..., description="Contenido del archivo en base64")
    size: int = Field(..., description="Tamaño del archivo en bytes")

class QueryWithDocumentsRequest(BaseModel):
    """Esquema para consulta con documentos adjuntos."""
    query: str = Field(..., description="Texto de la consulta del usuario")
    documents: Optional[List[DocumentData]] = Field(None, description="Lista de documentos adjuntos")