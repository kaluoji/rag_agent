# test_rag_agent.py

import pytest
from types import SimpleNamespace
from agents.ai_expert_v1 import (
    ai_expert,
    AIDeps,
    retrieve_relevant_documentation,
    retrieve_detailed_information,
    generate_compliance_report,
)
from utils.supabase_wrapper import SupabaseClientWrapper  # Nueva importación
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai import models

pytestmark = pytest.mark.asyncio


##############################
# Definir FakeAIDeps (para tests)
##############################
from pydantic import BaseModel
from typing import Any

class FakeAIDeps(BaseModel):
    supabase: Any  # Usamos Any para evitar problemas de validación
    openai_client: Any

    class Config:
        arbitrary_types_allowed = True

##############################
# Fixtures para tests
##############################
@pytest.fixture
def fake_deps():
    return FakeAIDeps(
        supabase=SupabaseClientWrapper(is_test=True).get_client(),
        openai_client=FakeOpenAIClient()
    )

# Aquí, en lugar de un dict, usamos SimpleNamespace para usage, con atributo 'requests'
@pytest.fixture
def fake_ctx(fake_deps):
    return RunContext(
        deps=fake_deps,
        model=TestModel(),
        usage=FakeUsage(),  # Usar la clase con el método incr
        prompt="Texto de prueba"
    )

##############################
# Implementaciones Falsas (Mocks)
##############################

class FakeSupabaseClient:

    def __init__(self, *args, **kwargs):
        # Puedes omitir la llamada al __init__ real o simularla según sea necesario
        pass

    def rpc(self, function_name, params):
        class FakeRPC:
            def execute(self):
                return SimpleNamespace(
                    data=[
                        {
                            "title": "Documento Dummy 1",
                            "summary": "Resumen Dummy 1",
                            "content": "Contenido dummy 1.",
                            "metadata": {"source": "TestDB"},
                            "literal_quote": "Cita dummy 1"
                        },
                        {
                            "title": "Documento Dummy 2",
                            "summary": "Resumen Dummy 2",
                            "content": "Contenido dummy 2.",
                            "metadata": {"source": "TestDB"},
                            "literal_quote": "Cita dummy 2"
                        },
                    ]
                )
        return FakeRPC()

class FakeEmbeddings:
    async def create(self, model, input):
        fake_embedding = [0.1] * 1536
        fake_item = SimpleNamespace(embedding=fake_embedding)
        fake_response = SimpleNamespace(data=[fake_item])
        return fake_response

from openai import AsyncOpenAI

class FakeOpenAIClient(AsyncOpenAI):
    def __init__(self):
        # No llamamos al __init__ real para evitar requerir parámetros
        self.embeddings = FakeEmbeddings()

class FakeUsage:
    def __init__(self):
        self.total_tokens = 0
        self.requests = 0
        self.request_tokens = 0    # Nuevo atributo
        self.response_tokens = 0   # Nuevo atributo
        self.details = {}          # Si el framework lo requiere

    def incr(self, usage, requests):
        # Supongamos que usage es un dict con tokens (ajusta según lo que espera el agente)
        self.total_tokens += getattr(usage, "total_tokens", 0)
        self.request_tokens += getattr(usage, "request_tokens", 0)
        self.response_tokens += getattr(usage, "response_tokens", 0)
        self.requests += requests

##############################
# Tests de las herramientas del agente
##############################

@pytest.mark.asyncio
async def test_retrieve_relevant_documentation(fake_ctx):
    user_query = "Consulta dummy sobre normativa"
    result = await retrieve_relevant_documentation(fake_ctx, user_query)
    assert "Documento Dummy 1" in result
    assert "Contenido dummy 1." in result

@pytest.mark.asyncio
async def test_retrieve_detailed_information_standard(fake_ctx):
    user_query = "Consulta detallada de normativa"
    result = await retrieve_detailed_information(fake_ctx, user_query, detail_level="standard")
    assert "Documento Dummy 1" in result
    assert "Contenido dummy 1." in result
    assert "<blockquote>" not in result

@pytest.mark.asyncio
async def test_retrieve_detailed_information_extended(fake_ctx):
    user_query = "Consulta detallada de normativa"
    result = await retrieve_detailed_information(fake_ctx, user_query, detail_level="extended")
    # Se espera que en el modo extendido aparezcan etiquetas HTML para citas y referencias.
    assert "Documento Dummy 1" in result
    assert "<strong>Summary:</strong>" in result
    assert "<blockquote>" in result
    # Aseguramos que se incluya la cita; aquí ajustamos el formato para que la cita aparezca sin blockquote,
    # o bien se verifique el contenido que se espera. Por ejemplo, si usamos:
    # <strong>Cita:</strong> <em>Cita dummy 1</em>
    # podemos ajustar la aserción:
    assert "Cita dummy 1" in result or "Cita dummy 1" in result.lower()

@pytest.mark.asyncio
async def test_generate_compliance_report(fake_ctx):
    compliance_output = "Resultado dummy del agente de Compliance."
    result = await generate_compliance_report(fake_ctx, compliance_output, template=None)
    assert "# Informe de Compliance" in result


@pytest.mark.asyncio
async def test_agent_run_with_test_model(fake_deps):
    # Permitir solicitudes de modelo en este test
    models.ALLOW_MODEL_REQUESTS = True
    with ai_expert.override(model=TestModel()):
        user_query = "¿Cuáles son las últimas actualizaciones normativas?"
        response = await ai_expert.run(user_query, deps=fake_deps)
        assert response.data is not None
