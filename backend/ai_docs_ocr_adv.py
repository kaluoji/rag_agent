import os
import json
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import pytesseract
import pdfplumber  # Reemplazado PyMuPDF con pdfplumber
from PIL import Image, ImageEnhance, ImageFilter
import tldextract
from dateutil import parser
from io import BytesIO
import atexit
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
from markitdown import MarkItDown

# Importar ProcessPoolExecutor para tareas intensivas en CPU
from concurrent.futures import ProcessPoolExecutor

# Importar biblioteca para rate limiting
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

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

# Añadir esta clase para gestionar rate limiting
class OpenAIRateLimiter:
    """
    Implementa un limitador de tasa para las llamadas a la API de OpenAI.
    Extrae el tiempo de espera sugerido de los mensajes de error y espera
    ese tiempo exacto antes de reintentar.
    """
    def __init__(self, rpm_limit=450):  # Usar 450 en lugar de 500 para tener margen
        self.rpm_limit = rpm_limit
        self.request_timestamps = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Espera si es necesario para no exceder el límite de RPM."""
        async with self.lock:
            now = time.time()
            # Eliminar timestamps más antiguos que 60 segundos
            self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
            
            if len(self.request_timestamps) >= self.rpm_limit:
                # Calcular cuánto tiempo esperar
                oldest = min(self.request_timestamps)
                wait_time = 60 - (now - oldest) + random.uniform(0.1, 0.5)  # Añadir jitter
                logging.info(f"Limitando tasa: esperando {wait_time:.2f} segundos")
                await asyncio.sleep(wait_time)
                
                # Limpiar timestamps antiguos nuevamente después de esperar
                now = time.time()
                self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
            
            # Registrar esta solicitud
            self.request_timestamps.append(now)

# Inicializar el limitador de tasa global
openai_limiter = OpenAIRateLimiter()

# Decorador para controlar tasa y reintentos
@retry(
    retry=retry_if_exception_type((TimeoutError, Exception)),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
async def rate_limited_openai_call(func, *args, **kwargs):
    """
    Ejecuta una función de llamada a la API de OpenAI con control de tasa
    y reintentos automáticos en caso de error.
    Extrae el tiempo de espera sugerido de los mensajes de error 429.
    """
    await openai_limiter.wait_if_needed()
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            logging.warning(f"OpenAI rate limit alcanzado: {e}")
            
            # Intentar extraer el tiempo de espera recomendado
            import re
            wait_time_match = re.search(r'Please try again in (\d+\.\d+)s', error_str)
            
            if wait_time_match:
                wait_time = float(wait_time_match.group(1))
                # Añadir un pequeño margen
                wait_time = wait_time + 0.5
                logging.info(f"Esperando {wait_time:.2f} segundos según lo recomendado por OpenAI")
                await asyncio.sleep(wait_time)
            else:
                # Si no podemos extraer el tiempo, esperamos un tiempo razonable
                await asyncio.sleep(random.uniform(2.0, 5.0))
        raise

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
# Función para dividir el texto en fragmentos - Semantic Chunking
# -------------------------------------------------------------------
async def semantic_chunk_text(text: str, chunk_size: int = 800, min_chunk_size: int = 500, max_chunks: int = 50, overlap_size: int = 75) -> List[Dict]:
    """
    Divide el texto en fragmentos semánticamente coherentes usando clustering.
    Incluye superposición entre chunks del mismo cluster para mantener contexto.
    
    Args:
        text: El texto a dividir
        chunk_size: Tamaño objetivo para cada chunk final
        min_chunk_size: Tamaño mínimo para considerar un chunk válido
        max_chunks: Número máximo de chunks a crear
        overlap_size: Cantidad de caracteres de superposición entre chunks
        
    Returns:
        Lista de chunks con metadatos de clustering
    """
    # Paso 1: División inicial en párrafos como unidades básicas
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []
    
    # Para textos muy cortos, usamos el método simple
    if len(text) < chunk_size * 2:
        return [{"text": chunk, "cluster_id": -1, "cluster_size": 1} for chunk in chunk_text(text, chunk_size)]
    
    # Paso 2: Obtener embeddings para cada párrafo
    batch_size = 20
    all_embeddings = await batch_get_embeddings(paragraphs, batch_size)
    
    # Convertir a array de numpy para clustering
    embeddings_array = np.array(all_embeddings)
    
    # Paso 3: Clustering
    num_clusters = min(max(len(paragraphs) // 3, 2), max_chunks)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Paso 4: Agrupar párrafos por cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((i, paragraphs[i]))
    
    # Paso 5: Crear chunks finales con superposición
    final_chunks_with_metadata = []
    
    for label, items in clusters.items():
        # Ordenamos los párrafos por su posición original
        items.sort(key=lambda x: x[0])
        
        # Crear chunks dentro de este cluster
        current_chunk = ""
        previous_paragraphs = []  # Almacenar los últimos párrafos para overlap
        overlap_text = ""
        
        for _, paragraph in items:
            # Si añadir este párrafo excede el tamaño y ya tenemos contenido suficiente
            if len(current_chunk) + len(paragraph) > chunk_size and len(current_chunk) >= min_chunk_size:
                # Guardar el chunk actual
                final_chunks_with_metadata.append({
                    "text": current_chunk.strip(),
                    "cluster_id": int(label),
                    "cluster_size": len(items),
                    "has_overlap": len(overlap_text) > 0
                })
                
                # Crear overlap para el siguiente chunk
                overlap_text = ""
                # Tomar los últimos párrafos hasta alcanzar approximate_overlap_size
                total_len = 0
                for p in reversed(previous_paragraphs):
                    if total_len + len(p) <= overlap_size:
                        overlap_text = p + "\n\n" + overlap_text
                        total_len += len(p) + 4  # +4 para \n\n
                    else:
                        # Si el párrafo es demasiado grande, tomar solo el final
                        if total_len == 0:  # Si es el primer párrafo y ya es grande
                            overlap_size_for_this_p = min(overlap_size, len(p))
                            overlap_text = p[-overlap_size_for_this_p:] + "\n\n" + overlap_text
                        break
                
                # Empezar nuevo chunk con el overlap
                current_chunk = overlap_text + paragraph
                previous_paragraphs = [paragraph]
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                previous_paragraphs.append(paragraph)
                # Mantener solo los últimos N párrafos para el overlap
                if len(previous_paragraphs) > 5:  # Ajustar según tus necesidades
                    previous_paragraphs = previous_paragraphs[-5:]
        
        # Añadir el último chunk si tiene contenido
        if current_chunk and len(current_chunk) >= min_chunk_size:
            final_chunks_with_metadata.append({
                "text": current_chunk.strip(),
                "cluster_id": int(label),
                "cluster_size": len(items),
                "has_overlap": len(overlap_text) > 0
            })
    
    # Fallback si no se crearon chunks válidos
    if not final_chunks_with_metadata:
        return [{"text": chunk, "cluster_id": -1, "cluster_size": 1, "has_overlap": False} 
                for chunk in chunk_text(text, chunk_size)]
    
    return final_chunks_with_metadata

def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """
    Divide el texto en fragmentos de tamaño similar.
    Método básico para cuando el semantic chunking falla.
    
    Args:
        text: El texto a dividir
        chunk_size: Tamaño objetivo para cada chunk
        
    Returns:
        Lista de chunks de texto
    """
    # Dividir por párrafos para preservar la estructura
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Si añadir este párrafo excede el tamaño del chunk y ya tenemos contenido
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Añadir el último chunk si tiene contenido
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Función para calcular similitud entre chunks (útil para verificar la calidad del clustering)
async def calculate_chunk_similarities(chunks: List[str]) -> Tuple[np.ndarray, List[Any]]:
    """
    Calcula la matriz de similitud coseno entre todos los chunks.
    
    Args:
        chunks: Lista de chunks de texto
        
    Returns:
        Tupla con (matriz de similitud, embeddings)
    """
    embeddings = []
    for chunk in chunks:
        emb = await get_embedding(chunk)
        embeddings.append(emb)
    
    emb_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(emb_array)
    
    return similarity_matrix, embeddings

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
        "For the summary: Give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.\n"
        "Keep both title and summary concise but informative.\n\n"
        "<document>\n"
        "{{WHOLE_DOCUMENT}}\n"
        "</document>\n"
        "Here is the chunk we want to situate within the whole document\n"
        "<chunk>\n"
        "{{CHUNK_CONTENT}}\n"
        "</chunk>\n"
        "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
    )
    try:
        async def call_api():
            return await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Identifier: {identifier}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
        
        response = await rate_limited_openai_call(call_api)
        return json.loads(response.choices[0].message.content)
    except RetryError as e:
        logging.error(f"Error después de múltiples intentos al obtener título y resumen: {e}")
        return {"title": "Error procesando el título", "summary": "Error procesando el resumen"}
    except Exception as e:
        logging.error(f"Error al obtener título y resumen: {e}")
        return {"title": "Error procesando el título", "summary": "Error procesando el resumen"}

async def get_embedding(text: str) -> List[float]:
    """Obtiene el embedding del texto usando OpenAI."""
    try:
        async def call_api():
            return await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
        
        response = await rate_limited_openai_call(call_api)
        return response.data[0].embedding
    except RetryError as e:
        logging.error(f"Error después de múltiples intentos al obtener embedding: {e}")
        return [0] * 1536  # Vector nulo en caso de error
    except Exception as e:
        logging.error(f"Error al obtener embedding: {e}")
        return [0] * 1536  # Vector nulo en caso de error

async def batch_get_embeddings(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """
    Obtiene embeddings para múltiples textos en un solo batch real,
    aprovechando la capacidad de la API de OpenAI para procesar múltiples
    textos en una sola llamada.
    
    Args:
        texts: Lista de textos para obtener embeddings
        batch_size: Tamaño máximo de batch para cada llamada a la API
        
    Returns:
        Lista de embeddings (un embedding por texto)
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            async def call_api():
                # Usar una sola llamada para múltiples textos
                return await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
            
            response = await rate_limited_openai_call(call_api)
            
            # Ordenar los embeddings según el índice original
            sorted_embeddings = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_embeddings]
            all_embeddings.extend(batch_embeddings)
            
            # Breve pausa entre batches para evitar sobrecargar la API
            if i + batch_size < len(texts):
                await asyncio.sleep(0.5)
                
        except RetryError as e:
            logging.error(f"Error después de múltiples intentos al obtener embeddings en batch: {e}")
            # Fallback: rellenar con vectores nulos
            null_embeddings = [[0] * 1536 for _ in range(len(batch))]
            all_embeddings.extend(null_embeddings)
        except Exception as e:
            logging.error(f"Error al obtener embeddings en batch: {e}")
            # Fallback: rellenar con vectores nulos
            null_embeddings = [[0] * 1536 for _ in range(len(batch))]
            all_embeddings.extend(null_embeddings)
    
    return all_embeddings



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
        async def call_api():
            return await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}
                ]
            )
        
        response = await rate_limited_openai_call(call_api)
        return response.choices[0].message.content.strip()
    except RetryError as e:
        logging.error(f"Error después de múltiples intentos al obtener la categoría: {e}")
        return "Otros"
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
        async def call_api():
            return await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}
                ]
            )
        
        response = await rate_limited_openai_call(call_api)
        return response.choices[0].message.content.strip()
    except RetryError as e:
        logging.error(f"Error después de múltiples intentos al obtener palabras clave: {e}")
        return "Otros"
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

async def process_chunk(chunk_with_metadata: Dict, chunk_number: int, identifier: str) -> ProcessedChunk:
    """
    Procesa un fragmento de texto con su metadata de cluster.
    """
    # Extraer el texto y la información del cluster
    chunk_text = chunk_with_metadata["text"] if isinstance(chunk_with_metadata, dict) else chunk_with_metadata
    cluster_id = chunk_with_metadata.get("cluster_id", -1) if isinstance(chunk_with_metadata, dict) else -1
    cluster_size = chunk_with_metadata.get("cluster_size", 1) if isinstance(chunk_with_metadata, dict) else 1
    
    # Obtener título/resumen y embedding en paralelo (son independientes)
    extracted_task = asyncio.create_task(get_title_and_summary(chunk_text, identifier))
    embedding_task = asyncio.create_task(get_embedding(chunk_text))
    
    # Esperar a que terminen estas tareas
    extracted = await extracted_task
    embedding = await embedding_task
    
    # Obtener metadata simple sin llamadas a API
    date = extract_date_from_url(identifier)
    summary = extracted.get('summary', '')
    source = await get_source(identifier)
    
    # Obtener categoría y keywords secuencialmente para controlar el rate limit
    category = await get_category(summary)
    keywords = await extract_keywords(summary)
    
    metadata = {
        "chunk_size": len(chunk_text),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "source_identifier": identifier,
        "date": date,
        "category": category,
        "keywords": keywords,
        "source": source,
        # Añadir la información del cluster a los metadatos
        "cluster_id": cluster_id,
        "cluster_size": cluster_size
    }
    return ProcessedChunk(
        url=identifier,
        chunk_number=chunk_number,
        title=extracted.get('title', ''),
        summary=extracted.get('summary', ''),
        content=chunk_text,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """
    Inserta el fragmento procesado en la tabla 'visa_mastercard_v7' de Supabase.
    Si falla, guarda los datos localmente para procesamiento posterior.
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
        result = supabase.table("visa_mastercard_v7").insert(data).execute()
        logging.info(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        logging.error(f"Error al insertar el fragmento: {e}")
        
        # Guardar datos localmente como respaldo
        try:
            os.makedirs("pending_chunks", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_url = "".join(c if c.isalnum() else "_" for c in chunk.url)[:50]
            file_path = f"pending_chunks/{safe_url}_{chunk.chunk_number}_{timestamp}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "url": chunk.url,
                    "chunk_number": chunk.chunk_number,
                    "title": chunk.title,
                    "summary": chunk.summary,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding
                }, f, default=str)
            
            logging.info(f"Datos guardados en {file_path} para procesamiento posterior")
            return {"status": "local", "file": file_path}
        except Exception as e2:
            logging.error(f"Error al guardar datos localmente: {e2}")
            return None
        return None

# -------------------------------------------------------------------
# Funciones para OCR
# -------------------------------------------------------------------
async def convert_to_markdown(text: str) -> str:
    """
    Convierte el texto extraído mediante OCR a formato Markdown
    para preservar su estructura original, utilizando MarkItDown y post-procesamiento.
    
    Args:
        text: Texto extraído mediante OCR
        
    Returns:
        Texto formateado en Markdown
    """
    from markitdown import MarkItDown
    import tempfile
    import os
    import re
    
    # Guardar el texto en un archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(text)
        temp_path = temp_file.name
    
    try:
        logging.info("Iniciando conversión a Markdown con MarkItDown")
        
        # Inicializar MarkItDown (con manejo de diferentes posibilidades de parámetros)
        try:
            # Primero intentamos sin parámetros adicionales
            md = MarkItDown()
            logging.info("MarkItDown inicializado correctamente sin parámetros")
        except Exception as e1:
            logging.warning(f"Primer intento de inicialización falló: {e1}")
            try:
                # Si falla, intentamos con use_plugins
                md = MarkItDown(use_plugins=True)
                logging.info("MarkItDown inicializado correctamente con use_plugins=True")
            except Exception as e2:
                logging.warning(f"Segundo intento de inicialización falló: {e2}")
                # Último intento
                md = MarkItDown(plugins_enabled=True)
                logging.info("MarkItDown inicializado correctamente con plugins_enabled=True")
        
        # Convertir el archivo a Markdown
        result = md.convert(temp_path)
        
        # Mejorar el formato del Markdown generado
        improved_content = post_process_markdown(result.text_content)
        logging.info("Conversión y post-procesamiento de Markdown completados con éxito")
        
        return improved_content
    
    except Exception as e:
        logging.error(f"Error en la conversión a Markdown: {e}")
        # En caso de error, usar el método de respaldo
        return basic_markdown_conversion(text)
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)

def post_process_markdown(content):
    """
    Mejora el formato del contenido Markdown, especialmente para tablas.
    
    Args:
        content: Contenido Markdown generado por MarkItDown
        
    Returns:
        Contenido Markdown mejorado
    """
    lines = content.split('\n')
    output_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        # Detectar posibles líneas de tabla (contienen múltiples '|')
        if '|' in line and line.count('|') >= 2:
            # Si no estábamos en una tabla, comenzamos una nueva
            if not in_table:
                in_table = True
                table_rows = [line]
            else:
                # Añadir línea a la tabla existente
                table_rows.append(line)
        else:
            # Si no es una línea de tabla pero estábamos en una tabla
            if in_table:
                # Procesar la tabla acumulada
                processed_table = format_table(table_rows)
                output_lines.extend(processed_table)
                in_table = False
                table_rows = []
            
            # Añadir la línea normal
            output_lines.append(line)
    
    # Si quedó una tabla al final del documento
    if in_table:
        processed_table = format_table(table_rows)
        output_lines.extend(processed_table)
    
    return '\n'.join(output_lines)

def format_table(table_rows):
    """
    Formatea correctamente una tabla Markdown.
    
    Args:
        table_rows: Lista de líneas que forman una tabla
        
    Returns:
        Lista de líneas con la tabla formateada correctamente
    """
    if not table_rows:
        return []
    
    # Limpiar espacios y formatear filas
    cleaned_rows = []
    for row in table_rows:
        # Normalizar separadores de tabla
        cleaned_row = re.sub(r'\s*\|\s*', ' | ', row.strip())
        if cleaned_row.startswith('| '):
            cleaned_row = cleaned_row
        else:
            cleaned_row = '| ' + cleaned_row
            
        if cleaned_row.endswith(' |'):
            cleaned_row = cleaned_row
        else:
            cleaned_row = cleaned_row + ' |'
            
        cleaned_rows.append(cleaned_row)
    
    if len(cleaned_rows) < 2:
        # Añadir encabezado si solo hay una fila
        header = cleaned_rows[0]
        separator = '| ' + ' | '.join(['---' for _ in range(header.count('|')-1)]) + ' |'
        return [header, separator]
    
    # Si hay múltiples filas pero no hay separador después del encabezado
    if not all(c == '|' or c == ' ' or c == '-' for c in cleaned_rows[1].replace('|', '').replace(' ', '').replace('-', '')):
        separator = '| ' + ' | '.join(['---' for _ in range(cleaned_rows[0].count('|')-1)]) + ' |'
        cleaned_rows.insert(1, separator)
    
    return cleaned_rows

def basic_markdown_conversion(text):
    """
    Método de respaldo para convertir a markdown en caso de fallar la conversión principal.
    Usa técnicas básicas de procesamiento de texto.
    
    Args:
        text: Texto extraído mediante OCR
        
    Returns:
        Texto formateado básico en Markdown
    """
    # Dividir el texto en líneas
    lines = text.split('\n')
    markdown_lines = []
    
    # Variables de control para detectar estructuras
    in_list = False
    in_table = False
    in_code_block = False
    
    # Patrones para detectar elementos
    header_pattern = re.compile(r'^([A-Z0-9][A-Z0-9\s]{0,40})$')
    bullet_pattern = re.compile(r'^\s*[•\-\*]\s')
    numbered_list_pattern = re.compile(r'^\s*(\d+[\.\)]\s)')
    table_row_pattern = re.compile(r'.*\|\s*.*\|.*')
    
    # Procesar línea por línea
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Detectar encabezados de página (--- Página X ---)
        if line.startswith('--- Página'):
            markdown_lines.append('\n## ' + line + '\n')
            continue
            
        # Línea vacía
        if not line.strip():
            in_list = False
            if in_table:
                in_table = False
            markdown_lines.append('')
            continue
            
        # Detectar tablas
        if '|' in line and table_row_pattern.match(line):
            if not in_table:
                in_table = True
                # Si estamos comenzando una tabla, agregar una fila de encabezado y separador
                if i > 0 and '|' not in lines[i-1]:
                    # Contar columnas y crear encabezado automático
                    cols = line.count('|') + 1
                    header = '| ' + ' | '.join([f'Columna {j+1}' for j in range(cols)]) + ' |'
                    separator = '| ' + ' | '.join(['---' for j in range(cols)]) + ' |'
                    markdown_lines.append(header)
                    markdown_lines.append(separator)
            # Formatear fila de tabla adecuadamente
            cells = line.split('|')
            formatted_line = '| ' + ' | '.join([cell.strip() for cell in cells]) + ' |'
            markdown_lines.append(formatted_line)
            continue
            
        # Detectar listas con viñetas
        if bullet_pattern.match(line):
            in_list = True
            markdown_lines.append(line)
            continue
            
        # Detectar listas numeradas
        if numbered_list_pattern.match(line):
            in_list = True
            markdown_lines.append(line)
            continue
            
        # Detectar encabezados
        if header_pattern.match(line.strip()) and len(line.strip()) < 60:
            next_line = lines[i+1].strip() if i+1 < len(lines) else ""
            # Si la siguiente línea está vacía, es probablemente un encabezado
            if not next_line or len(next_line) < 3:
                markdown_lines.append(f'\n## {line.strip()}\n')
                continue
        
        # Detectar secciones que parecen código (líneas con indentación consistente)
        if line.startswith('    ') and not in_code_block:
            in_code_block = True
            markdown_lines.append('```')
            markdown_lines.append(line)
            continue
        elif in_code_block and not line.startswith('    '):
            in_code_block = False
            markdown_lines.append('```')
        
        # Si estamos en un bloque de código, mantener la indentación
        if in_code_block:
            markdown_lines.append(line)
            continue
            
        # Línea normal
        markdown_lines.append(line)
    
    # Cerrar cualquier bloque de código abierto
    if in_code_block:
        markdown_lines.append('```')
    
    return '\n'.join(markdown_lines)


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """
    Aplica preprocesamiento a la imagen: conversión a escala de grises, ajuste de contraste y afilado.
    """
    pil_img = pil_img.convert('L')
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    return pil_img



# -------------------------------------------------------------------
# Función global para procesar una página (para ProcessPoolExecutor)
# -------------------------------------------------------------------
async def async_extract_text_from_page(page, page_number: int, process_pool=None) -> str:
    """
    Extrae el texto de una única página de un PDF de forma asíncrona.
    Usa pdfplumber para extraer texto, con OCR como respaldo.
    """
    loop = asyncio.get_running_loop()
    try:
        # Extraer texto con pdfplumber
        text = page.extract_text() or ""
        
        # Si hay poco texto, realizar OCR en la imagen de la página
        if len(text.strip()) < 50:
            img = page.to_image()
            img_pil = img.original
            
            # Ejecutar el preprocesamiento y OCR en el process pool
            async def process_ocr():
                # Preprocesamiento de imagen
                img_processed = preprocess_image(img_pil)
                # Realizar OCR
                ocr_text = pytesseract.image_to_string(img_processed, config="--psm 6")
                return ocr_text
            
            # Ejecutar OCR en el executor para no bloquear
            text = await loop.run_in_executor(process_pool, lambda: pytesseract.image_to_string(preprocess_image(img_pil), config="--psm 6"))
        
        return text + f"\n\n--- Página {page_number + 1} ---\n\n"
    except Exception as e:
        logging.error(f"Error en async_extract_text_from_page para página {page_number}: {e}")
        return ""

async def async_extract_text_from_pdf(file_path: str, process_pool=None) -> str:
    """
    Abre el PDF con pdfplumber y procesa cada página en paralelo.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            tasks = []
            for i, page in enumerate(pdf.pages):
                tasks.append(async_extract_text_from_page(page, i, process_pool))
            texts = await asyncio.gather(*tasks)
            return "".join(texts)
    except Exception as e:
        logging.error(f"Error en async_extract_text_from_pdf para {file_path}: {e}")
        return ""

async def async_extract_text(file_path: str, process_pool=None) -> str:
    """
    Extrae el texto de un archivo de forma asíncrona.
    Para PDFs, utiliza pdfplumber con la extracción paralela a nivel de páginas.
    Para imágenes, utiliza la función sincrónica original.
    """
    loop = asyncio.get_running_loop()
    if file_path.lower().endswith('.pdf'):
        return await async_extract_text_from_pdf(file_path, process_pool)
    else:
        return await loop.run_in_executor(process_pool, extract_text_from_file, file_path)

# -------------------------------------------------------------------
# Función para procesar y almacenar documento OCR - MODIFICADA
# -------------------------------------------------------------------
async def process_and_store_ocr_document(file_path: str):
    """
    Para un archivo OCR:
      1. Extrae el texto usando OCR.
      2. Convierte el texto a formato Markdown.
      3. Divide el texto en fragmentos usando semantic chunking.
      4. Procesa cada fragmento (título/resumen, embedding, etc.) en batches pequeños.
      5. Inserta los fragmentos en Supabase.
    """
    logging.info(f"Procesando OCR para el documento: {file_path}")
    ocr_text = await async_extract_text(file_path, process_pool)
    
    # Convertir el texto extraído a formato Markdown
    logging.info(f"Convirtiendo a Markdown el texto extraído de: {file_path}")
    markdown_text = await convert_to_markdown(ocr_text)
    logging.info(f"Conversión a Markdown completada para: {file_path}")
    
    # Usar semantic chunking en lugar de chunking simple
    chunks_with_metadata = await semantic_chunk_text(markdown_text)
    logging.info(f"Documento {file_path} dividido en {len(chunks_with_metadata)} fragmentos usando semantic chunking.")
    
    # Procesar chunks en batches pequeños para evitar demasiadas peticiones simultáneas
    batch_size = 5
    for i in range(0, len(chunks_with_metadata), batch_size):
        batch = chunks_with_metadata[i:i+batch_size]
        
        # Procesar este batch
        logging.info(f"Procesando batch de chunks {i}-{i+len(batch)-1} de {len(chunks_with_metadata)}")
        tasks = [process_chunk(chunk, i+idx, file_path) for idx, chunk in enumerate(batch)]
        processed_chunks = await asyncio.gather(*tasks)
        
        # Insertar los chunks procesados
        insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
        await asyncio.gather(*insert_tasks)
        
        # Breve pausa entre batches para evitar sobrecargar la API
        if i + batch_size < len(chunks_with_metadata):
            await asyncio.sleep(3)
    
    logging.info(f"Documento OCR {file_path} procesado, convertido a Markdown y almacenado.")

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

async def ocr_parallel(file_paths: List[str], max_concurrent: int = 2):  # Reducir de 5 a 2
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