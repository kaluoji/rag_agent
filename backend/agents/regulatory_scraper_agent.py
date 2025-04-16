from __future__ import annotations as _annotations

import logging
import httpx
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin
import re

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from supabase import Client
from bs4 import BeautifulSoup

from app.core.config import settings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización del modelo
model = OpenAIModel(
    model_name=settings.llm_model,
    api_key=settings.openai_api_key,
    
)

class Publication(BaseModel):
    """Modelo para una publicación regulatoria."""
    title: str = Field(..., description="Título de la publicación")
    url: str = Field(..., description="URL de la publicación")
    type: str = Field(..., description="Tipo de publicación (ej: Guía, Recomendación, Opinión)")
    publication_date: datetime = Field(..., description="Fecha de publicación")
    issuer: str = Field(..., description="Organismo emisor")
    summary: Optional[str] = Field(None, description="Resumen del contenido")

class ScraperDeps(BaseModel):
    """Dependencias necesarias para el agente scraper."""
    supabase: Client
    openai_client: AsyncOpenAI
    http_client: httpx.AsyncClient

    class Config:
        arbitrary_types_allowed = True

class ScrapingResult(BaseModel):
    """Resultado del scraping de una fuente."""
    source_url: str
    publications: List[Publication]
    total_publications: int
    last_update: datetime = Field(default_factory=datetime.now)

system_prompt = """
Eres un agente especializado en extraer y analizar información regulatoria de fuentes oficiales europeas. Tu objetivo es procesar páginas web de autoridades regulatorias y extraer información estructurada sobre nuevas publicaciones, guías y normativas.

Principales responsabilidades:
1. Extraer información relevante de las páginas web proporcionadas
2. Estructurar la información en un formato consistente
3. Identificar y categorizar diferentes tipos de publicaciones
4. Asegurar la precisión en la extracción de fechas y metadatos

Al procesar cada página:
- Identifica claramente el título, tipo de documento, fecha y emisor
- Extrae URLs completas y válidas
- Categoriza correctamente el tipo de publicación
- Valida que la información extraída sea coherente y completa

Herramientas disponibles:
- fetch_page: Obtiene el contenido HTML de una URL
- parse_content: Analiza el contenido HTML y extrae información estructurada
- store_publications: Almacena las publicaciones en la base de datos

Utiliza las herramientas de manera secuencial para procesar cada fuente y asegura que la información extraída sea precisa y útil para el sistema de compliance.
"""

regulatory_scraper = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=ScraperDeps,
    result_type=ScrapingResult,
    retries=3,  # Aumentar número de reintentos
    model_settings={
        "messages": [
            {
                "role": "assistant",
                "content": system_prompt
            }
        ]
    }
)

@regulatory_scraper.tool
async def fetch_page(ctx: RunContext[ScraperDeps], url: str) -> str:
    """Obtiene el contenido HTML de una URL específica."""
    try:
        response = await ctx.deps.http_client.get(url)
        response.raise_for_status()
        
        # Extraer solo el contenido relevante para reducir tokens
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Eliminar elementos innecesarios que consumen tokens
        for element in soup.find_all(['script', 'style', 'meta', 'link', 'footer', 'header', 'nav']):
            element.decompose()
            
        # Encontrar el contenedor principal
        main_content = soup.find(['main', 'article', 'div', 'section'], 
                               class_=['content', 'main-content', 'document-library'])
        
        if main_content:
            return str(main_content)
        return str(soup.body) if soup.body else str(soup)
        
    except Exception as e:
        logger.error(f"Error fetching page {url}: {e}")
        return ""

@regulatory_scraper.tool
async def parse_content(
    ctx: RunContext[ScraperDeps], 
    html_content: str, 
    base_url: str
) -> List[Publication]:
    """Analiza el contenido HTML y extrae publicaciones estructuradas."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        publications = []
        
        # Limitar el análisis a los elementos más recientes
        items = soup.find_all(['article', 'div', 'tr'], 
                            class_=['publication', 'document', 'item'])[:30]  # Limitar a 30 items
        
        for item in items:
            try:
                # Limpiar y extraer texto
                for tag in item.find_all(['script', 'style']):
                    tag.decompose()
                
                title_elem = item.find(['h2', 'h3', 'h4', 'a'], class_=['title', 'heading'])
                if not title_elem:
                    title_elem = item.find('a')
                
                date_elem = item.find(class_=['date', 'publication-date', 'time'])
                if not date_elem:
                    date_elem = item.find(['time', 'span'], string=re.compile(r'\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}'))
                
                if not all([title_elem, date_elem]):
                    continue
                
                # Extraer y limpiar URL
                url = title_elem.get('href') if title_elem.name == 'a' else item.find('a')['href']
                url = urljoin(base_url, url)
                
                # Parsear fecha con manejo de múltiples formatos
                try:
                    pub_date = parse_date(date_elem.text.strip())
                except:
                    pub_date = datetime.now()
                
                # Crear publicación con datos limitados
                pub = Publication(
                    title=title_elem.text.strip()[:200],  # Limitar longitud
                    url=url,
                    type=determine_publication_type(item),
                    publication_date=pub_date,
                    issuer=determine_issuer(base_url)
                )
                publications.append(pub)
                
            except Exception as e:
                logger.warning(f"Error parsing individual publication: {e}")
                continue
        
        return publications[:20]  # Retornar máximo 20 publicaciones
        
    except Exception as e:
        logger.error(f"Error in parse_content: {e}")
        return []

def parse_date(date_text: str) -> datetime:
    """
    Función auxiliar para parsear fechas en múltiples formatos.
    """
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    
    return datetime.now()

@regulatory_scraper.tool
async def store_publications(
    ctx: RunContext[ScraperDeps],
    publications: List[Publication]
) -> bool:
    """Almacena las publicaciones en la base de datos."""
    try:
        # Convertir publicaciones a formato para inserción
        records = []
        for pub in publications:
            try:
                record = {
                    "title": pub.title.strip(),
                    "url": pub.url,
                    "type": pub.type,
                    "publication_date": pub.publication_date.isoformat(),
                    "issuer": pub.issuer,
                    "summary": pub.summary if pub.summary else "",  # Valor por defecto para campo opcional
                    "created_at": datetime.now().isoformat()
                }
                
                # Verificar solo los campos requeridos
                required_fields = ["title", "url", "type", "publication_date", "issuer"]
                if all(record[field] for field in required_fields):
                    records.append(record)
                else:
                    missing_fields = [f for f in required_fields if not record[f]]
                    logger.warning(f"Registro incompleto. Faltan campos: {missing_fields}")
                    
            except Exception as e:
                logger.error(f"Error preparando registro: {e}")
                continue

        if not records:
            logger.warning("No hay registros válidos para insertar")
            return False

        # Insertar en Supabase usando upsert para evitar duplicados
        result = ctx.deps.supabase.table("regulatory_publications").upsert(
            records,
            on_conflict="url"  # Usar URL como clave única
        ).execute()

        if result.data:
            logger.info(f"Se almacenaron exitosamente {len(result.data)} publicaciones")
            return True
        else:
            logger.error("No se retornaron datos de la operación de inserción")
            return False

    except Exception as e:
        logger.error(f"Error almacenando publicaciones: {e}")
        return False

def determine_publication_type(item_soup: BeautifulSoup) -> str:
    """Determina el tipo de publicación basado en el contenido HTML."""
    type_indicators = {
        'guideline': 'Guía',
        'recommendation': 'Recomendación',
        'opinion': 'Opinión',
        'consultation': 'Consulta',
        'stress': 'Test de Estrés'
    }
    
    text_content = item_soup.text.lower()
    for indicator, pub_type in type_indicators.items():
        if indicator in text_content:
            return pub_type
    
    return "Otros"

def determine_issuer(url: str) -> str:
    """Determina el emisor basado en la URL."""
    issuer_mapping = {
        'eiopa.europa.eu': 'EIOPA',
        'esma.europa.eu': 'ESMA',
        'esrb.europa.eu': 'ESRB',
        'edpb.europa.eu': 'EDPB'
    }
    
    for domain, issuer in issuer_mapping.items():
        if domain in url:
            return issuer
    
    return "Desconocido"

# Función auxiliar para ejecutar el scraping
async def run_regulatory_scraping(deps: ScraperDeps, urls: List[str]) -> List[ScrapingResult]:
    """
    Ejecuta el scraping en todas las URLs proporcionadas.
    
    Args:
        deps: Dependencias necesarias para el scraping
        urls: Lista de URLs a procesar
    
    Returns:
        List[ScrapingResult]: Resultados del scraping
    """
    results = []
    
    for url in urls:
        try:
            # Primero intentamos obtener el contenido de la página
            html_content = await fetch_page(RunContext(deps), url)
            if not html_content:
                logger.warning(f"No se pudo obtener contenido de {url}")
                continue
                
            # Procesamos el contenido
            publications = await parse_content(RunContext(deps), html_content, url)
            
            # Almacenamos las publicaciones
            if publications:
                success = await store_publications(RunContext(deps), publications)
                if success:
                    results.append(ScrapingResult(
                        source_url=url,
                        publications=publications,
                        total_publications=len(publications)
                    ))
                    logger.info(f"Procesadas exitosamente {len(publications)} publicaciones de {url}")
                else:
                    logger.error(f"Error al almacenar publicaciones de {url}")
            else:
                logger.warning(f"No se encontraron publicaciones en {url}")
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            continue
    
    return results