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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Rastrear múltiples URLs en paralelo con un límite de concurrencia."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        viewport_width=1280,
        viewport_height=720,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        text_mode=True  # Optimizar para extracción de texto
    )
        
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=True,
        css_selector=".c-article",  
        wait_until="networkidle",  # Esperar hasta que la red esté inactiva
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
                    print("Contenido obtenido:")
                    print(result.markdown_v2.raw_markdown)
                    #await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Falló: {url} - Error: {result.error_message}")
            
        # Procesar todas las URLs en paralelo con concurrencia limitada
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()


def get_specific_urls() -> List[str]:
    """Retornar una lista de URLs específicas para rastrear."""
    return [
        "https://www.eleconomista.com.mx/tags/mastercard-29756",
        "https://www.eleconomista.com.mx/buscar/?q=visa/"
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
