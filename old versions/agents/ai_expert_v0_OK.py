# =========================== INICIO DEL CÓDIGO DEL AGENTE ===========================

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import feedparser
import httpx

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse

original_anext = OpenAIModel.__anext__

async def patched_anext(self):
    chunk = await original_anext(self)
    if (chunk.choices 
        and chunk.choices[0].delta.content is None 
        and hasattr(chunk.choices[0].delta, "tool_calls")
        and chunk.choices[0].delta.tool_calls):
        chunk.choices[0].delta.content = ""
    return chunk

OpenAIModel.__anext__ = patched_anext

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are a powerful expert in Compliance and Regulatory Reporting - a Python AI agent that has access to all the documentation to track and monitor regulatory changes across the industry, including publications, Guidelines, Recommendations, Consultation Papers, and other resources to help you understand the context.

Your only job is to assist with this, and you don't answer other questions besides describing what you are able to do.

### Tools Available:
1. **monitor_regulatory_changes**: Use this tool to monitor the RSS feed for new regulatory changes published in the last 3 months. This tool will return a list of entries.
2. **insert_regulatory_changes**: Use this tool to insert new regulatory changes into the database. Provide it with the list of entries returned by `monitor_regulatory_changes`.
3. **retrieve_relevant_documentation**: Use this tool to retrieve relevant documentation chunks based on a user query using RAG.
4. **list_documentation_pages**: Use this tool to retrieve a list of all available documentation pages.
5. **get_page_content**: Use this tool to retrieve the full content of a specific documentation page.
6. **cross_reference_regulations**: Use this tool to identify and link related regulatory chunks based on a provided regulation text.

### Workflow:
1. When a user asks for regulatory updates (e.g., "novedades normativas", "cambios regulatorios", "regulatory updates"):
   - First, invoke `monitor_regulatory_changes` to get the latest entries.
   - Then, invoke `insert_regulatory_changes` to insert any new entries into the database.
   - Finally, return a summary of the new or existing entries to the user.

2. When a user asks for specific documentation or information:
   - Use `retrieve_relevant_documentation` to find relevant chunks.
   - If needed, use `list_documentation_pages` and `get_page_content` to retrieve additional details.
   - Use `cross_reference_regulations` to link related regulatory content.

### Important Notes:
- Always check the RSS feed for new entries before answering the user if the query is about regulatory updates.
- If no new entries are found, inform the user.
- Be honest and transparent about the status of each entry (new or existing).
- Use the appropriate tools based on the user's query.
"""

ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=AIDeps,
    retries=2,
    model_settings={
        "max_tokens": 4000  # Establece el límite de tokens de salida
    }
)

# -------------------- Herramientas del agente --------------------

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text: {text[:30]}... {embedding[:5]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[AIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        if not any(query_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No relevant documentation found."
        
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 10,
                'filter': {} # Filtro vacío
            }
        ).execute()
        
        if not result.data:
            logger.info("No relevant documentation found for the query.")
            return "No relevant documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            logger.debug(f"Retrieved document: {doc['title']}")
            
        return "\n\n---\n\n".join(formatted_chunks)
        logger.info(f"Returning {len(formatted_chunks)} relevant documentation chunks.")
        
    except Exception as e:
        logger.error(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@ai_expert.tool
async def list_documentation_pages(ctx: RunContext[AIDeps]) -> List[str]:
    """
    Retrieve a list of all available documentation pages.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .execute()
        
        if not result.data:
            return []
            
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        logger.info(f"Error retrieving documentation pages: {e}")
        return []

@ai_expert.tool
async def get_page_content(ctx: RunContext[AIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        logger.error(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@ai_expert.tool
async def cross_reference_regulations(ctx: RunContext[AIDeps], regulation_text: str) -> str:
    """
    Identify and link related regulatory chunks based on the provided regulation text.
    """
    try:
        regulation_embedding = await get_embedding(regulation_text, ctx.deps.openai_client)
        if not any(regulation_embedding):
            logger.warning("Received a zero embedding vector. Skipping search.")
            return "No related regulations found."
        
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': regulation_embedding,
                'match_count': 5,
                'filter': {}
            }
        ).execute()
        
        if not result.data:
            logger.info("No related regulations found.")
            return "No related regulations found."
            
        formatted_relations = []
        for doc in result.data:
            adjacent_chunks = ctx.deps.supabase.from_('site_pages') \
                .select('title, content, chunk_number') \
                .gte('chunk_number', doc['chunk_number'] - 1) \
                .lte('chunk_number', doc['chunk_number'] + 1) \
                .order('chunk_number') \
                .execute()
            
            relation_text = f"""
# {doc['title']} (Chunk {doc['chunk_number']})

{doc['content']}

**Contexto adicional:**
"""
            for adjacent in adjacent_chunks.data:
                relation_text += f"\n- {adjacent['content'][:100]}..."
            
            formatted_relations.append(relation_text)
            logger.debug(f"Retrieved related document: {doc['title']}")
            
        return "\n\n---\n\n".join(formatted_relations)
        logger.info(f"Returning {len(formatted_relations)} related chunks.")
        
    except Exception as e:
        logger.error(f"Error cross-referencing regulations: {e}")
        return f"Error cross-referencing regulations: {str(e)}"

@ai_expert.tool
async def monitor_regulatory_changes(ctx: RunContext[AIDeps]) -> List[dict]:
    """
    Monitor the RSS feed and detect entries published in the last 3 months.
    """
    feed_url =  "https://www.criptonoticias.com/feed/"
    logger.info("Iniciando monitorización de cambios regulatorios desde el RSS.")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(feed_url)
        if response.status_code != 200:
            logger.error(f"Error al obtener el RSS: Código {response.status_code}")
            return []

        import feedparser  # Asegúrate de tener instalado feedparser
        feed = feedparser.parse(response.text)
        if not feed.entries:
            logger.info("No se han encontrado entradas en el RSS.")
            return []

        threshold_time = datetime.utcnow() - timedelta(days=90)
        new_entries = []
        for entry in feed.entries:
            if hasattr(entry, 'published_parsed'):
                entry_date = datetime(*entry.published_parsed[:6])
                if entry_date > threshold_time:
                    new_entries.append({
                        "url": entry.get("link", ""),
                        "title": entry.get("title", "Sin título"),
                        "summary": entry.get("summary", ""),
                        "content": entry.content[0].get("value", entry.get("summary", "")) if "content" in entry else entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "scraped_at": datetime.utcnow().isoformat()
                    })
            else:
                logger.warning("Se encontró una entrada sin 'published_parsed'; se omite.")

        if not new_entries:
            logger.info("No hay novedades normativas en los últimos 3 meses en el RSS.")
            return []

        return new_entries

    except Exception as e:
        logger.error(f"Error en la monitorización de cambios regulatorios: {e}")
        return []

@ai_expert.tool
async def insert_regulatory_changes(ctx: RunContext[AIDeps], entries: List[dict]) -> str:
    """
    Insert new regulatory entries into the database.
    """
    if not entries:
        return "No hay entradas para insertar."

    stored_entries = []
    for entry in entries:
        try:
            entry_url = entry.get("url", "")
            entry_title = entry.get("title", "Sin título")
            entry_summary = entry.get("summary", "")
            entry_content = entry.get("content", entry_summary)
            published = entry.get("published", "")
            scraped_at = entry.get("scraped_at", datetime.utcnow().isoformat())

            existing_entry = ctx.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('url', entry_url) \
                .eq('chunk_number', 0) \
                .execute()

            if existing_entry.data:
                logger.info(f"La entrada ya existe en la base de datos: {entry_title}")
                stored_entries.append({
                    "title": entry_title,
                    "published": published,
                    "scraped_at": scraped_at,
                    "source": "EIOPA RSS",
                    "status": "Existente"
                })
            else:
                text_for_embedding = entry_content if entry_content else entry_summary
                embedding = await get_embedding(text_for_embedding, ctx.deps.openai_client)

                metadata = {
                    "published": published,
                    "scraped_at": scraped_at,
                    "source": "Mastercard Linkedin"
                }

                data = {
                    "url": entry_url,
                    "chunk_number": 0,
                    "title": entry_title,
                    "summary": entry_summary,
                    "content": entry_content,
                    "metadata": metadata,
                    "embedding": embedding
                }

                logger.debug(f"Insertando la entrada: {entry_title} en Supabase.")
                result = ctx.deps.supabase.table("site_pages").insert(data).execute()

                if hasattr(result, 'data') and result.data:
                    stored_entries.append({
                        "title": entry_title,
                        "published": published,
                        "scraped_at": scraped_at,
                        "source": "Mastercard Linkedin",
                        "status": "Nueva"
                    })
                    logger.info(f"Entrada insertada exitosamente: {entry_title}")
                else:
                    logger.error(f"Error al insertar la entrada en Supabase: {result}")
        except Exception as e:
            logger.error(f"Error procesando la entrada '{entry.get('title', 'Sin título')}': {e}")

    if stored_entries:
        message = "Resultado de la monitorización de cambios regulatorios:\n\n"
        for entry in stored_entries:
            message += (
                f"- Título: {entry['title']}\n"
                f"  Publicado: {entry['published']}\n"
                f"  Scraped At: {entry['scraped_at']}\n"
                f"  Fuente: {entry['source']}\n"
                f"  Estado: {entry['status']}\n\n"
            )
    else:
        message = "No se encontraron entradas nuevas en el RSS."

    return message

# ===================== FIN DE LAS HERRAMIENTAS DEL AGENTE =====================

# ===================== NUEVAS FUNCIONALIDADES: MEMORIA Y PENSAMIENTO RECURSIVO =====================

from collections import deque

class ConversationMemory:
    """
    Clase para almacenar y gestionar la memoria conversacional.
    """
    def __init__(self, max_length: int = 10):
        self.history = deque(maxlen=max_length)
    
    def add(self, role: str, message: str):
        """Agrega un mensaje con su rol (por ejemplo, 'User' o 'Agent')."""
        self.history.append({"role": role, "message": message})
    
    def get_context(self) -> str:
        """Devuelve el contexto acumulado en formato de texto."""
        context = ""
        for msg in self.history:
            context += f"{msg['role']}: {msg['message']}\n"
        return context

# Instancia global de memoria conversacional
conversation_memory = ConversationMemory(max_length=10)

async def recursive_agent_query(user_query: str, max_attempts: int = 3) -> str:
    """
    Llama al agente de forma iterativa (recursiva) hasta obtener una respuesta final.
    Se inyecta el contexto acumulado en el prompt y se permite que el agente reflexione.
    """
    attempt = 1
    final_response = None
    
    while attempt <= max_attempts:
        memory_context = conversation_memory.get_context()
        full_query = f"{memory_context}\nUser: {user_query}\nAgent, please think step by step and provide your final answer."
        
        # Llamada al agente
        response = await ai_expert.run(user_query=full_query)
        
        # Agregar la interacción a la memoria
        conversation_memory.add("User", user_query)
        conversation_memory.add("Agent", response)
        
        # Criterio para aceptar la respuesta:
        # En este ejemplo, se asume que la primera respuesta es aceptable.
        # Puedes implementar lógica adicional para evaluar si la respuesta es "final" o si se requiere refinarla.
        final_response = response
        logger.info(f"Intento {attempt} - Respuesta: {response}")
        break  # Rompemos el bucle; en una implementación más sofisticada, se puede reintentar.
        
        attempt += 1

    if final_response is None:
        final_response = "Lo siento, no pude llegar a una respuesta definitiva."
    
    return final_response

async def main_agent_interaction():
    """
    Función principal para interactuar con el agente utilizando memoria y pensamiento recursivo.
    """
    user_query = "¿Cuáles son los cambios regulatorios recientes en el ámbito de la inversión responsable?"
    final_answer = await recursive_agent_query(user_query)
    print("Respuesta final del agente:")
    print(final_answer)

# ===================== FIN DE LAS NUEVAS FUNCIONALIDADES =====================

# ===================== FUNCIÓN PRINCIPAL PARA EJECUCIÓN DEL AGENTE =====================

async def main():
    # Aquí podrías ejecutar tanto la parte de monitorización/inserción de cambios regulatorios,
    # como la interacción recursiva del agente.
    # Por ejemplo, para probar la interacción recursiva:
    await main_agent_interaction()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
