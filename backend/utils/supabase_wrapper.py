# utils/supabase_wrapper.py

import os
import logging
from typing import Optional, Any, Dict, List
from types import SimpleNamespace

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSupabaseClient:
    """
    Cliente mock para Supabase que imita la interfaz necesaria sin depender de la implementación real.
    """
    def __init__(self):
        self.url = "https://mock-supabase-url.com"
        self.key = "mock-key"
        # Agregamos estos atributos para imitar mejor a SyncClient
        self.auth = self.auth
        self.rest = None
        self.realtime = None
        self.postgrest = None
        self.storage = None
        logger.info("Inicializando MockSupabaseClient para testing")

    def rpc(self, function_name: str, params: dict) -> Any:
        """Simula llamadas RPC retornando datos de prueba"""
        logger.debug(f"Mock RPC call - función: {function_name}, params: {params}")
        
        class MockRPCResponse:
            def execute(self):
                return SimpleNamespace(
                    data=[{
                        "title": "Documento Dummy para Testing",
                        "summary": "Este es un resumen de prueba para testing.",
                        "content": "Contenido dummy para simular respuestas de la base de datos.",
                        "metadata": {
                            "source": "TestDB",
                            "created_at": "2024-02-16",
                            "type": "test_document"
                        },
                        "literal_quote": "Esta es una cita de prueba para testing."
                    }]
                )
        
        return MockRPCResponse()

    def auth(self):
        """Simula el endpoint de autenticación"""
        return SimpleNamespace(current_user=None)

class SupabaseClientWrapper:
    """
    Wrapper para manejar las dependencias de Supabase tanto en producción como en testing.
    """
    
    def __init__(self, is_test: bool = False, config: Optional[Dict[str, str]] = None):
        self.is_test = is_test
        self._client = None
        self.config = config or {}
        
        if not is_test:
            try:
                from supabase import create_client
                
                supabase_url = self.config.get('SUPABASE_URL') or os.getenv('SUPABASE_URL')
                supabase_key = self.config.get('SUPABASE_KEY') or os.getenv('SUPABASE_KEY')
                
                if not all([supabase_url, supabase_key]):
                    logger.warning("Credenciales de Supabase no encontradas. Usando cliente mock.")
                    self._client = MockSupabaseClient()
                else:
                    self._client = create_client(supabase_url, supabase_key)
                    logger.info("Cliente Supabase real inicializado correctamente.")
                    
            except ImportError:
                logger.warning("Supabase client no disponible. Usando versión mock para testing.")
                self._client = MockSupabaseClient()
        else:
            logger.info("Inicializando cliente mock para testing.")
            self._client = MockSupabaseClient()

    def get_client(self) -> Any:
        """
        Retorna el cliente de Supabase (real o mock).
        """
        if self._client is None:
            raise RuntimeError("El cliente Supabase no ha sido inicializado correctamente.")
        return self._client

    def is_dummy_client(self) -> bool:
        """
        Verifica si el cliente actual es un mock client.
        """
        return isinstance(self._client, MockSupabaseClient)