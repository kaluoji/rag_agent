import asyncio
import httpx
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client
import logging
from rich import print as rprint
from rich.table import Table
from datetime import datetime

# Añadir el directorio raíz del proyecto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.config import settings
from agents.regulatory_scraper_agent import regulatory_scraper, ScraperDeps, run_regulatory_scraping

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# URLs a procesar
REGULATORY_URLS = [
    "https://www.eiopa.europa.eu/document-library/guidelines_en",
    "https://www.esma.europa.eu/databases-library/esma-library",
    "https://www.esrb.europa.eu/mppa/recommendations/html/index.en.html",
    "https://www.esrb.europa.eu/mppa/stress/html/index.en.html",
    "https://www.esrb.europa.eu/mppa/opinions/html/index.en.html",
    "https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en",
    "https://edpb.europa.eu/our-work-tools/general-guidance/public-consultations-our-guidance_en"
]

def create_results_table(results):
    """Crear una tabla con los resultados del scraping."""
    table = Table(title="Resultados del Scraping Regulatorio")
    
    table.add_column("Fuente", style="cyan")
    table.add_column("Publicaciones", style="magenta")
    table.add_column("Última Actualización", style="green")
    
    for result in results:
        table.add_row(
            result.source_url,
            str(result.total_publications),
            result.last_update.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    return table

async def process_and_display_publications(deps, urls):
    """Procesar las URLs y mostrar los resultados de forma detallada."""
    print("\n[bold blue]Iniciando scraping regulatorio...[/bold blue]")
    
    for url in urls:
        try:
            print(f"\n[yellow]Procesando[/yellow]: {url}")
            
            # Añadir delay entre requests para evitar rate limits
            await asyncio.sleep(2)
            
            # Obtener el contenido
            html_content = await deps.http_client.get(
                url,
                timeout=30.0,  # Añadir timeout
                follow_redirects=True  # Añadir seguimiento de redirecciones
            )
            if html_content.status_code != 200:
                print(f"[red]Error obteniendo contenido de {url}[/red]")
                continue
                
            print(f"[green]✓[/green] Contenido obtenido correctamente")
            
            # Procesar publicaciones con manejo de errores mejorado
            try:
                publications = await regulatory_scraper.run(
                    f"Process regulatory content from {url}",
                    deps=deps
                )
                
                if publications and hasattr(publications, 'data') and publications.data and publications.data.publications:
                    print(f"\n[green]Encontradas {len(publications.data.publications)} publicaciones[/green]")
                    
                    # Mostrar solo un resumen
                    print("\n[cyan]Resumen de publicaciones encontradas:[/cyan]")
                    print(f"Total: {len(publications.data.publications)}")
                    if publications.data.publications:
                        print("Última publicación:")
                        pub = publications.data.publications[0]
                        print(f"  Título: {pub.title[:100]}...")
                        print(f"  Fecha: {pub.publication_date}")
                
            except Exception as e:
                print(f"[red]Error procesando publicaciones: {str(e)}[/red]")
                logger.exception("Error detallado:")
                
        except Exception as e:
            print(f"[red]Error general procesando {url}: {str(e)}[/red]")
            continue
            
        print("\n[green]---[/green]")

async def main():
    # Inicializar clientes
    try:
        supabase = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        
        # Verificar conexión
        test_query = supabase.table('regulatory_publications').select("count", count='exact').limit(1).execute()
        logger.info("Conexión a Supabase exitosa")
        
    except Exception as e:
        logger.error(f"Error al conectar con Supabase: {e}")
        raise
        
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    async with httpx.AsyncClient() as http_client:
        # Crear dependencias
        deps = ScraperDeps(
            supabase=supabase,
            openai_client=openai_client,
            http_client=http_client
        )
        
        # Ejecutar y mostrar resultados
        await process_and_display_publications(deps, REGULATORY_URLS)

if __name__ == "__main__":
    asyncio.run(main())