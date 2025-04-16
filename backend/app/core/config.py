import os
import logging
from pathlib import Path
from pydantic_settings import BaseSettings

# Create a logger for this module
logger = logging.getLogger(__name__)

# Calcula BASE_DIR para apuntar a la raíz del proyecto (rag_agent)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Configuración original
    llm_model: str = "gpt-4o-mini"  
    tokenizer_model: str = "gpt-4o-mini"
    openai_api_key: str
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    embedding_model: str = "text-embedding-3-small"
    
    # Nuevas configuraciones para FastAPI
    PROJECT_NAME: str = "VISA Mastercard Expert API"
    PROJECT_DESCRIPTION: str = "API para consultas y generación de reportes sobre normativas de VISA y Mastercard"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]
    
    # Configuración de directorios
    REPORTS_DIR: str = os.path.join(BASE_DIR, "output", "reports")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    MAX_REPORT_GENERATION_TIME: int = 300

    class Config:
        # Usamos la ruta absoluta al .env en la raíz del proyecto
        env_file = os.path.join(BASE_DIR, ".env")
        env_file_encoding = "utf-8"

    def __init__(self, **data):
        super().__init__(**data)
        # Verificar variables críticas
        logger.info(f"supabase_url configurado: {'Sí' if self.supabase_url else 'No'}")
        logger.info(f"supabase_key configurado: {'Sí' if self.supabase_key else 'No'}")
        logger.info(f"openai_api_key configurado: {'Sí' if self.openai_api_key else 'No'}")

settings = Settings()