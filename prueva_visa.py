import asyncio
import logging
from config.config import settings
from openai import AsyncOpenAI
from supabase import create_client
from agents.ai_expert_v1 import debug_run_agent, AIDeps

# Configuración de logging agresiva
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def test_direct_query():
    """
    Prueba directa del agente de compliance sin pasar por el orquestador.
    Esto nos permite verificar si las herramientas se están invocando correctamente.
    """
    logger.info("Iniciando prueba directa del agente de compliance")
    
    # Inicializar clientes
    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    # Crear dependencias
    deps = AIDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    
    # Consulta que definitivamente debería invocar retrieve_relevant_documentation
    query = "¿Cuáles son las reglas de VISA y Mastercard para el procesamiento de transacciones internacionales?"
    
    logger.info(f"Enviando consulta de prueba: {query}")
    
    # Llamar directamente al agente de compliance
    try:
        response = await debug_run_agent(query, deps=deps)
        logger.info(f"Respuesta recibida: {response.data[:100]}...")  # Mostrar solo los primeros 100 caracteres
        logger.info(f"Uso de tokens: {response.usage()}")
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}")

if __name__ == "__main__":
    logger.info("Iniciando script de prueba")
    asyncio.run(test_direct_query())
    logger.info("Script de prueba finalizado")