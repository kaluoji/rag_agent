import os
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import tldextract
from dateutil import parser
from io import BytesIO
import atexit
import logging

# Importar ProcessPoolExecutor para tareas intensivas en CPU
from concurrent.futures import ProcessPoolExecutor

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cargar variables de entorno
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------
# Inicializar clientes globales
# ---------------------------
# Estos clientes DEBEN definirse a nivel de módulo para que estén disponibles en todas las funciones.
from openai import AsyncOpenAI
from supabase import create_client, Client

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Definir el executor globalmente (se usará para las tareas intensivas en CPU)
process_pool = ProcessPoolExecutor(max_workers=5)
atexit.register(lambda: process_pool.shutdown(wait=True))

# -------------------------------------------------------------------
# Definición de la clase ProcessedChunk
# -------------------------------------------------------------------
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# -------------------------------------------------------------------
# Función para dividir el texto en fragmentos
# -------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 1400) -> List[str]:
    """
    Divide el texto en fragmentos respetando bloques de código y párrafos.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)
    
    return chunks

# -------------------------------------------------------------------
# Funciones de procesamiento (Título/Resumen, Embeddings, etc.)
# -------------------------------------------------------------------
async def get_title_and_summary(chunk: str, identifier: str) -> Dict[str, str]:
    """
    Extrae título y resumen usando OpenAI.
    El parámetro 'identifier' puede ser una URL o el path del archivo.
    """
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks in the same language as the chunk.\n"
        "Return a JSON object with 'title' and 'summary' keys.\n"
        "For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.\n"
        "For the summary: Create a concise summary of the main points in this chunk.\n"
        "Keep both title and summary concise but informative."
    )
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Identifier: {identifier}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Error al obtener título y resumen: {e}")
        return {"title": "Error procesando el título", "summary": "Error procesando el resumen"}

async def get_embedding(text: str) -> List[float]:
    """Obtiene el embedding del texto usando OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error al obtener embedding: {e}")
        return [0] * 1536  # Vector nulo en caso de error

def extract_date_from_url(identifier: str) -> str:
    """
    Intenta extraer una fecha del 'identifier' (URL o path). 
    Si no se encuentra, devuelve la fecha actual.
    """
    try:
        parsed = urlparse(identifier)
        path_segments = parsed.path.split('/')
        for segment in path_segments:
            try:
                date = parser.parse(segment, fuzzy=False)
                return date.isoformat()
            except (ValueError, OverflowError):
                continue
    except Exception as e:
        logging.error(f"Error al extraer fecha: {e}")
    return datetime.now(timezone.utc).isoformat()

async def get_category(chunk: str) -> str:
    """
    Clasifica el fragmento en una categoría y subcategoría predefinida usando GPT-4.
    """
    system_prompt = (
        "Eres un modelo de IA que clasifica fragmentos de texto en categorías y subcategorías predefinidas.\n"
        "La clasificación se organiza así:\n\n"
        "Categoría: Sostenibilidad\n"
        "Subcategoría: ESG\n"
        "Subcategoría: SFDR\n"
        "Subcategoría: Green MIFID\n"
        "Subcategoría: Métricas e informes de sostenibilidad\n"
        "Subcategoría: Estrategias de inversión responsable\n\n"
        "Categoría: Riesgos Financieros\n"
        "Subcategoría: Riesgo de crédito\n"
        "Subcategoría: Riesgo de mercado\n"
        "Subcategoría: Riesgo de contraparte\n"
        "Subcategoría: Riesgo operacional\n"
        "Subcategoría: Gestión de riesgo de terceros\n\n"
        "Categoría: Regulación y Supervisión\n"
        "Subcategoría: PBC/FT (Prevención de Blanqueo de Capitales / Financiación del Terrorismo)\n"
        "Subcategoría: MiCA (Markets in Crypto-Assets)\n"
        "Subcategoría: Regulación IA\n"
        "Subcategoría: Supervisión bancaria\n"
        "Subcategoría: Protección del consumidor\n\n"
        "Categoría: Seguridad Financiera\n"
        "Subcategoría: Fraude\n"
        "Subcategoría: Know Your Customer (KYC)\n"
        "Subcategoría: Protección de datos\n"
        "Subcategoría: Ciberseguridad\n"
        "Subcategoría: Medios de pago\n\n"
        "Categoría: Reporting Regulatorio\n"
        "Subcategoría: FINREP/COREP\n"
        "Subcategoría: Reportes de liquidez\n"
        "Subcategoría: IFRS\n"
        "Subcategoría: Reporting de capital y solvencia\n"
        "Subcategoría: Reporting ESG\n\n"
        "Categoría: Tesorería\n"
        "Subcategoría: Gestión de liquidez\n"
        "Subcategoría: Instrumentos de financiación\n"
        "Subcategoría: Control de pagos y cobros\n"
        "Subcategoría: Cobertura de riesgos de tipo de interés y tipo de cambio\n"
        "Subcategoría: Gestión de activos y pasivos a corto plazo\n\n"
        "A partir de esta lista, clasifica cada fragmento de texto en exactamente una categoría y una subcategoría (la que consideres más relevante)."
    )
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error al obtener la categoría: {e}")
        return "Otros"

async def extract_keywords(chunk: str) -> str:
    """
    Extrae dos palabras clave representativas del fragmento usando GPT-4.
    """
    system_prompt = (
        "Eres un modelo de IA que extrae palabras clave de fragmentos de texto.\n"
        "Para cada fragmento identifica y devuelve dos palabras clave que representan los temas principales del contenido."
    )
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error al obtener palabras clave: {e}")
        return "Otros"

async def get_source(identifier: str) -> str:
    """
    Devuelve la fuente a partir del identifier.
    Si es una URL se extrae el dominio; si es un file path se usa el nombre del archivo.
    """
    try:
        if identifier.startswith("http"):
            extracted = tldextract.extract(identifier)
            domain = f"{extracted.domain}.{extracted.suffix}"
            return domain
        else:
            return os.path.basename(identifier)
    except Exception as e:
        logging.error(f"Error al obtener la fuente: {e}")
        return "fuente_desconocida"

async def process_chunk(chunk: str, chunk_number: int, identifier: str) -> ProcessedChunk:
    """
    Procesa un fragmento de texto:
      - Extrae título y resumen.
      - Obtiene el embedding.
      - Extrae fecha, categoría y palabras clave.
      - Arma la metadata y retorna un objeto ProcessedChunk.
    """
    extracted = await get_title_and_summary(chunk, identifier)
    embedding = await get_embedding(chunk)
    date = extract_date_from_url(identifier)
    summary = extracted.get('summary', '')
    category = await get_category(summary)
    keywords = await extract_keywords(summary)
    source = await get_source(identifier)
    metadata = {
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "source_identifier": identifier,
        "date": date,
        "category": category,
        "keywords": keywords,
        "source": source
    }
    return ProcessedChunk(
        url=identifier,
        chunk_number=chunk_number,
        title=extracted.get('title', ''),
        summary=extracted.get('summary', ''),
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """
    Inserta el fragmento procesado en la tabla 'site_pages3' de Supabase.
    """
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
        logging.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        logging.error(f"Error al insertar el fragmento: {e}")
        return None

# -------------------------------------------------------------------
# Funciones para OCR
# -------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """
    Aplica preprocesamiento a la imagen: conversión a escala de grises, ajuste de contraste y afilado.
    """
    pil_img = pil_img.convert('L')
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    return pil_img

def extract_text_from_file(file_path: str) -> str:
    """
    Extrae el texto de un archivo mediante OCR.
    Soporta archivos PDF (convierte cada página a imagen) e imágenes.
    Para PDFs se utiliza PyMuPDF (fitz).
    """
    extracted_text = ""
    if file_path.lower().endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            for page_number in range(len(doc)):
                page = doc[page_number]
                zoom = 2.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pil_img = Image.open(BytesIO(img_data))
                pil_img = preprocess_image(pil_img)
                text = pytesseract.image_to_string(pil_img, config="--psm 6")
                extracted_text += text
                extracted_text += f"\n\n--- Página {page_number + 1} ---\n\n"
            doc.close()
        except Exception as e:
            logging.error(f"Error procesando PDF {file_path}: {e}")
    else:
        try:
            img = Image.open(file_path)
            img = preprocess_image(img)
            extracted_text = pytesseract.image_to_string(img, config="--psm 6")
        except Exception as e:
            logging.error(f"Error procesando imagen {file_path}: {e}")
    return extracted_text

# -------------------------------------------------------------------
# Función global para procesar una página (para ProcessPoolExecutor)
# -------------------------------------------------------------------
def process_page_sync(img_data: bytes, page_number: int) -> str:
    """
    Función de nivel global que recibe los bytes de la imagen de una página,
    realiza el preprocesamiento y aplica OCR.
    """
    try:
        pil_img = Image.open(BytesIO(img_data))
        pil_img = preprocess_image(pil_img)
        text = pytesseract.image_to_string(pil_img, config="--psm 6")
        return text + f"\n\n--- Página {page_number + 1} ---\n\n"
    except Exception as e:
        logging.error(f"Error procesando la página {page_number}: {e}")
        return ""

async def async_extract_text_from_page(page, page_number: int, process_pool=None) -> str:
    """
    Extrae el texto de una única página de un PDF de forma asíncrona.
    Separa la conversión a bytes (en el proceso principal) y el procesamiento (en el executor).
    """
    loop = asyncio.get_running_loop()
    zoom = 1.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return await loop.run_in_executor(process_pool, process_page_sync, img_data, page_number)

async def async_extract_text_from_pdf(file_path: str, process_pool=None) -> str:
    """
    Abre el PDF y procesa cada página en paralelo.
    """
    try:
        doc = fitz.open(file_path)
        tasks = []
        for i, page in enumerate(doc):
            tasks.append(async_extract_text_from_page(page, i, process_pool))
        texts = await asyncio.gather(*tasks)
        doc.close()
        return "".join(texts)
    except Exception as e:
        logging.error(f"Error en async_extract_text_from_pdf para {file_path}: {e}")
        return ""

async def async_extract_text(file_path: str, process_pool=None) -> str:
    """
    Extrae el texto de un archivo de forma asíncrona.
    Para PDFs, utiliza la extracción paralela a nivel de páginas.
    Para imágenes, utiliza la función sincrónica original.
    """
    loop = asyncio.get_running_loop()
    if file_path.lower().endswith('.pdf'):
        return await async_extract_text_from_pdf(file_path, process_pool)
    else:
        return await loop.run_in_executor(process_pool, extract_text_from_file, file_path)

# -------------------------------------------------------------------
# Función para procesar y almacenar documento OCR
# -------------------------------------------------------------------
async def process_and_store_ocr_document(file_path: str):
    """
    Para un archivo OCR:
      1. Extrae el texto usando OCR.
      2. Divide el texto en fragmentos.
      3. Procesa cada fragmento (título/resumen, embedding, etc.).
      4. Inserta los fragmentos en Supabase.
    """
    logging.info(f"Procesando OCR para el documento: {file_path}")
    ocr_text = await async_extract_text(file_path, process_pool)
    chunks = chunk_text(ocr_text)
    logging.info(f"Documento {file_path} dividido en {len(chunks)} fragmentos.")
    
    tasks = [process_chunk(chunk, i, file_path) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)
    
    logging.info(f"Documento OCR {file_path} procesado y almacenado.")

def get_ocr_file_paths() -> List[str]:
    """
    Retorna una lista de paths de archivos ubicados en la carpeta 'uploads_ocr'.
    Se consideran archivos PDF e imágenes.
    """
    folder = "uploads_ocr"
    if not os.path.exists(folder):
        logging.warning(f"La carpeta {folder} no existe.")
        return []
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff'))
    ]
    return file_paths

async def ocr_parallel(file_paths: List[str], max_concurrent: int = 5):
    """
    Procesa múltiples archivos OCR en paralelo, limitando la concurrencia.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_file(file_path: str):
        async with semaphore:
            try:
                await process_and_store_ocr_document(file_path)
            except Exception as e:
                logging.error(f"Error al procesar {file_path}: {e}")
    
    await asyncio.gather(*[process_file(fp) for fp in file_paths])

# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
async def main():
    file_paths = get_ocr_file_paths()
    if not file_paths:
        logging.info("No se encontraron documentos OCR para procesar.")
        return
    logging.info(f"Encontrados {len(file_paths)} documentos OCR para procesar.")
    await ocr_parallel(file_paths)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {e}")
