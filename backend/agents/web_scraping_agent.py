import logging
import datetime
from typing import List
from httpx import Client
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
import pandas as pd

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo simplificado - similar al de referencia
model = OpenAIModel("gpt-4", api_key="tu-api-key-aqui")

# ------------------------------
# 1) Modelos simplificados
# ------------------------------

class Regulation(BaseModel):
    """Modelo simplificado para una regulación"""
    title: str = Field(description="Title of the regulation")
    reference_number: str | None = Field(description="Reference number (e.g., EIOPA-BoS-20/600)", default=None)
    publication_date: str | None = Field(description="Publication date", default=None)
    document_url: str | None = Field(description="URL to the document", default=None)


class RegulationResults(BaseModel):
    """Resultado del scraping"""
    regulations: List[Regulation] = Field(description="List of regulations found")


# ------------------------------
# 2) Agente simplificado
# ------------------------------

web_scraping_agent = Agent(
    name="Web Scraping Agent",
    model=model,
    system_prompt="""
Your task is to extract regulatory information from a webpage.

Step 1: Fetch the text content from the URL using fetch_page_text()
Step 2: Parse the text and extract regulations with their:
   - Title
   - Reference number (if available)
   - Publication date (if available)
   - Document URL (if available)

Return the data as a structured list of regulations.
""",
    retries=2,
    result_type=RegulationResults,
    model_settings=ModelSettings(
        max_tokens=4000,
        temperature=0.1
    ),
)


# ------------------------------
# 3) Herramienta de scraping
# ------------------------------

@web_scraping_agent.tool_plain(retries=1)
def fetch_page_text(url: str) -> str:
    """
    Fetches and extracts text content from a webpage
    
    Args:
        url: The URL to scrape
        
    Returns:
        str: The text content of the page
    """
    logger.info(f"Fetching page: {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    }

    try:
        with Client(headers=headers, timeout=30) as client:
            response = client.get(url)
            
            if response.status_code != 200:
                return f"Error: Status code {response.status_code}"
            
            # Extraer solo el texto, como en el código de referencia
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text()
            
            # Limpiar el texto eliminando líneas vacías excesivas
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            logger.info(f"Successfully extracted text. Length: {len(clean_text)}")
            return clean_text
            
    except Exception as e:
        logger.error(f"Error fetching page: {e}")
        return f"Error: {str(e)}"


# ------------------------------
# 4) Función principal simple
# ------------------------------

def main():
    """Función principal simplificada"""
    
    # URL a scrapear
    url = "https://www.eiopa.europa.eu/document-library/guidelines_en"
    
    # Crear el prompt
    prompt = f"Please extract all regulations from this URL: {url}"
    
    try:
        # Ejecutar el agente de forma síncrona (como en el código de referencia)
        response = web_scraping_agent.run_sync(prompt)
        
        if response.data is None:
            print("No data returned from the model")
            return
        
        # Mostrar estadísticas de tokens
        print("-" * 50)
        print(f"Input tokens: {response.usage().request_tokens}")
        print(f"Output tokens: {response.usage().response_tokens}")
        print(f"Total tokens: {response.usage().total_tokens}")
        print("-" * 50)
        
        # Convertir a lista de diccionarios
        regulations_list = []
        for reg in response.data.regulations:
            regulations_list.append(reg.model_dump())
        
        # Crear DataFrame y guardar CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df = pd.DataFrame(regulations_list)
        filename = f"regulations_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Mostrar resumen
        print(f"\nTotal regulations found: {len(regulations_list)}")
        print(f"CSV saved as: {filename}")
        
        # Mostrar primeras 3 regulaciones
        print("\nFirst 3 regulations:")
        for i, reg in enumerate(response.data.regulations[:3]):
            print(f"\n{i+1}. {reg.title}")
            if reg.reference_number:
                print(f"   Reference: {reg.reference_number}")
            if reg.publication_date:
                print(f"   Date: {reg.publication_date}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()