import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import ollama
from dateutil import parser
from sklearn.feature_extraction.text import TfidfVectorizer
import tldextract
from tenacity import retry, wait_exponential, stop_after_attempt
from functools import lru_cache

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from aiolimiter import AsyncLimiter
import logging

logging.basicConfig(level=logging.DEBUG)  # O ajusta el nivel según necesites
logger = logging.getLogger(__name__)

# Define un limitador global para las llamadas a la API de OpenAI.
# Ajusta los parámetros según el límite de tu plan.
# Por ejemplo: 20 solicitudes por 60 segundos.
openai_chat_rate_limiter = AsyncLimiter(20, 60)


@lru_cache(maxsize=1)
def get_title_summary_system_prompt() -> str:
    """
    Retorna el prompt estático para extraer título y resumen de un fragmento.
    Se cachea para evitar reconstruirlo en cada llamada.
    """
    return (
        "You are an AI that extracts titles and summaries from documentation chunks in the same language as the chunk.\n"
        "Return a JSON object with 'title' and 'summary' keys.\n"
        "For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.\n"
        "For the summary: Create a concise summary of the main points in this chunk.\n"
        "Keep both title and summary concise but informative."
    )


@lru_cache(maxsize=1)
def get_agent_static_instructions() -> str:
    """
    Retorna las instrucciones estáticas para la consulta recursiva del agente.
    Se cachea para no incluirlas repetidamente en el prompt.
    """
    return (
        "Por favor, piensa paso a paso para responder a la siguiente consulta y genera una respuesta final clara.\n\n"
        "Explica tu razonamiento paso a paso y al final proporciona tu respuesta final."
    )


load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 1400) -> List[str]:
    """Divide el texto en fragmentos, respetando bloques de código y párrafos."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calcular la posición final
        end = start + chunk_size

        # Si estamos al final del texto, tomar lo que queda
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Intentar encontrar un límite de bloque de código primero (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # Si no hay bloque de código, intentar romper en un párrafo
        elif '\n\n' in chunk:
            # Encontrar el último salto de párrafo
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Solo romper si supera el 30% del tamaño del fragmento
                end = start + last_break

        # Si no hay salto de párrafo, intentar romper en una oración
        elif '. ' in chunk:
            # Encontrar el último punto de oración
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Solo romper si supera el 30% del tamaño del fragmento
                end = start + last_period + 1

        # Extraer el fragmento y limpiarlo
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Mover la posición inicial para el siguiente fragmento
        start = max(start + 1, end)

    return chunks


@retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=3), stop=stop_after_attempt(2))
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extraer título y resumen usando GPT-4."""
    system_prompt = get_title_summary_system_prompt ()

    try:
        async with openai_chat_rate_limiter:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Enviar los primeros 1000 caracteres para contexto
                ],
                response_format={ "type": "json_object" },
            
            )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error al obtener título y resumen: {e}")
        return {"title": "Error procesando el título", "summary": "Error procesando el resumen"}

import asyncio

@retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=3), stop=stop_after_attempt(2))
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Obtiene el embedding del texto usando OpenAI de forma robusta con reintentos."""
    try:
        async with openai_chat_rate_limiter:  # Control de rate limiting
            response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                
            )
        if not response.data or not hasattr(response.data[0], 'embedding'):
            raise ValueError("Respuesta de embedding vacía o inválida.")
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text: {text[:30]}... {embedding[:5]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text: {text[:30]}...; Error: {e}")
        raise e


def extract_date_from_url(url: str) -> str:
    """Extraer la fecha de la URL si está presente."""
    try:
        # Implementa una lógica específica según el formato de tus URLs
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.split('/')
        # Supongamos que la fecha está en el formato YYYY-MM-DD o similar
        for segment in path_segments:
            try:
                date = parser.parse(segment, fuzzy=False)
                return date.isoformat()
            except (ValueError, OverflowError):
                continue
    except Exception as e:
        print(f"Error al extraer la fecha de la URL: {e}")
    return datetime.now(timezone.utc).isoformat()  # Fecha por defecto si no se encuentra

@retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=3), stop=stop_after_attempt(2))
async def get_category(chunk: str) -> str:
    """Determinar la categoría del fragmento usando GPT-4."""
    system_prompt = """Eres un modelo de IA que clasifica fragmentos de texto en categorías y subcategorías predefinidas.
La clasificación se organiza así:

Categoría: Sostenibilidad
Subcategoría: ESG
Subcategoría: SFDR
Subcategoría: Green MIFID
Subcategoría: Métricas e informes de sostenibilidad
Subcategoría: Estrategias de inversión responsable

Categoría: Riesgos Financieros
Subcategoría: Riesgo de crédito
Subcategoría: Riesgo de mercado
Subcategoría: Riesgo de contraparte
Subcategoría: Riesgo operacional
Subcategoría: Gestión de riesgo de terceros

Categoría: Regulación y Supervisión
Subcategoría: PBC/FT (Prevención de Blanqueo de Capitales / Financiación del Terrorismo)
Subcategoría: MiCA (Markets in Crypto-Assets)
Subcategoría: Regulación IA
Subcategoría: Supervisión bancaria
Subcategoría: Protección del consumidor

Categoría: Seguridad Financiera
Subcategoría: Fraude
Subcategoría: Know Your Customer (KYC)
Subcategoría: Protección de datos
Subcategoría: Ciberseguridad
Subcategoría: Medios de pago

Categoría: Reporting Regulatorio
Subcategoría: FINREP/COREP
Subcategoría: Reportes de liquidez
Subcategoría: IFRS
Subcategoría: Reporting de capital y solvencia
Subcategoría: Reporting ESG

Categoría: Tesorería
Subcategoría: Gestión de liquidez
Subcategoría: Instrumentos de financiación
Subcategoría: Control de pagos y cobros
Subcategoría: Cobertura de riesgos de tipo de interés y tipo de cambio
Subcategoría: Gestión de activos y pasivos a corto plazo

A partir de esta lista, clasifica cada fragmento de texto en exactamente una categoría y una subcategoría (la que consideres más relevante). """

    try:
        async with openai_chat_rate_limiter:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}  
                ],
            
        )
        category = response.choices[0].message.content.strip()
        return category
    except Exception as e:
        print(f"Error al obtener la categoría: {e}")
        return "Otros"

@retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=3), stop=stop_after_attempt(2))
async def extract_keywords(chunk: str) -> str:
    """Determinar la categoría del fragmento usando GPT-4."""
    system_prompt = """Eres un modelo de IA que extrae palabras clave de fragmentos de texto.
    Para cada fragmento identifica y devuelve dos palabras clave que representan los temas principales del contenido. 
    """

    try:
        async with openai_chat_rate_limiter:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}  
                ],
            
            )
        category = response.choices[0].message.content.strip()
        return category
    except Exception as e:
        print(f"Error al obtener la categoría: {e}")
        return "Otros"


async def get_source(url: str) -> str:
    """Extrae el nombre de la fuente (dominio completo) de la URL."""
    try:
        extracted = tldextract.extract(url)
        # Reconstruir el dominio completo sin el subdominio
        domain = f"{extracted.domain}.{extracted.suffix}"
        return domain
    except Exception as e:
        print(f"Error al extraer la fuente de la URL: {e}")
        return "fuente_desconocida"


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Procesar un solo fragmento de texto."""
    # Obtener título y resumen
    extracted = await get_title_and_summary(chunk, url)
    
    # Obtener embedding
    embedding = await get_embedding(chunk, openai_client)

    # Obtener fecha
    date = extract_date_from_url(url)

    # Obtener categoría
    summary = extracted['summary']
    category = await get_category(summary)

    # Obtener palabras clave
    keywords = await extract_keywords(summary)

    # Obtener fuente
    source = await get_source(url)
    
    # Crear metadata
    metadata = {
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
        "date": date,
        "category": category,
        "keywords": keywords,
        "source": source
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insertar un fragmento procesado en Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages3").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error al insertar el fragmento: {e}")
        return None


async def process_and_store_document(url: str, markdown: str):
    """Procesar un documento y almacenar sus fragmentos en paralelo."""
    # Dividir en fragmentos
    chunks = chunk_text(markdown)
    
    # Procesar fragmentos en paralelo
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Almacenar fragmentos en paralelo
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 2):
    """Rastrear múltiples URLs en paralelo con un límite de concurrencia."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        viewport_width=1280,
        viewport_height=720,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        text_mode=True,  # Optimizar para extracción de texto
        
    )
    
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        wait_until="load",  
        scan_full_page=True,
        word_count_threshold=100  # Reducir el umbral de palabras para capturar más contenido
    )

    # Crear la instancia del rastreador
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Crear un semáforo para limitar la concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Rastreado exitosamente: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Falló: {url} - Error: {result.error_message}")
        
        # Procesar todas las URLs en paralelo con concurrencia limitada
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


def get_specific_urls() -> List[str]:
    """Retornar una lista de URLs específicas para rastrear."""
    return [
        "https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=uriserv%3AOJ.L_.2016.119.01.0001.01.SPA&toc=OJ%3AL%3A2016%3A119%3ATOC",
        "https://eur-lex.europa.eu/legal-content/ES/TXT/PDF/?uri=CELEX:32016R0679R(03)&from=ES",
        "https://eur-lex.europa.eu/legal-content/ES/TXT/PDF/?uri=CELEX:32016R0679R(02)&from=ES",
        "https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=uriserv%3AOJ.L_.2016.119.01.0089.01.SPA&toc=OJ%3AL%3A2016%3A119%3ATOC",
        "https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=uriserv%3AOJ.L_.2016.119.01.0132.01.SPA&toc=OJ%3AL%3A2016%3A119%3ATOC",
        "https://www.boe.es/buscar/pdf/1999/BOE-A-1999-23750-consolidado.pdf",
        "https://www.boe.es/buscar/act.php?id=BOE-A-2008-979",
        "https://www.boe.es/buscar/act.php?id=BOE-A-2018-16673",
        "https://www.boe.es/buscar/act.php?id=BOE-A-2021-8806",
        "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2019-3423",
        "https://www.aepd.es/guias/guia-cookies.pdf",
        "https://www.aepd.es/guias/faq-pruebas-concepto-verificacion-edad.pdf",
        "https://www.aepd.es/guias/guia-control-presencia-biometrico.pdf",
        "https://www.bcn.cl/leychile/navegar?idNorma=141599&idParte=8642684&idVersion=2026-12-01"      
        

    ]


async def main():
    # Definir las URLs específicas a rastrear
    urls = get_specific_urls()
    if not urls:
        print("No se encontraron URLs para rastrear")
        return
    
    print(f"Encontradas {len(urls)} URLs para rastrear")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
