from __future__ import annotations
from typing import Literal
import asyncio
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from datetime import datetime
import base64

import streamlit as st
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Importar únicamente las clases necesarias para mostrar mensajes
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

# Importar el orquestador en lugar del agente de compliance directamente
from agents.orchestrator_agent import process_query, OrchestratorDeps, AgentType
from agents.risk_assessment_agent import RiskAssessment

# Configuración de la página
st.set_page_config(
    page_title="AgentIA",
    layout="wide"
)

# Estilos CSS básicos y de personalización
st.markdown("""
<style>
    /* Estilo para el input de chat */
    .stTextInput > div > div > input {
        padding: 15px;
        border-radius: 15px;
        background-color: #f0f2f6;
        border: none;
    }
    /* Ocultar elementos predeterminados de Streamlit */
    .stDeployButton, #MainMenu, footer {
        display: none !important;
    }
    /* Estilos del sidebar */
    .sidebar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding: 10px 0;
    }
    /* Estilo para el modo experto */
    .risk-assessment {
        background-color: #f8f9fa;
        border-left: 4px solid #8B0D18;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .risk-area {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 4px;
        background-color: #fff;
        border: 1px solid #e0e0e0;
    }
    .risk-low { border-left: 4px solid #28a745; }
    .risk-medium { border-left: 4px solid #ffc107; }
    .risk-high { border-left: 4px solid #fd7e14; }
    .risk-critical { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Funciones para gestionar el estado de la sesión
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}  # {chat_id: messages}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chat_history[chat_id] = []
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.rerun()

def switch_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]
        st.rerun()

def get_chat_title(messages) -> str:
    """Extrae un título basado en el primer mensaje del usuario."""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if part.part_kind == 'user-prompt':
                    return part.content[:30] + "..." if len(part.content) > 30 else part.content
    return "Nueva conversación"

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        if chat_id == st.session_state.current_chat_id:
            st.session_state.current_chat_id = None
            st.session_state.messages = []
        st.rerun()

# Cargar variables de entorno y configurar clientes
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
logfire.configure(send_to_logfire='never')

# Utilidad para convertir archivos binarios (imágenes) a base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Función para mostrar una parte de un mensaje en la UI
def display_message_part(part):
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant", avatar=f"data:image/png;base64,{get_base64_of_bin_file('assets/logo.png')}"):
            st.markdown(part.content)

def display_risk_assessment(risk_data: RiskAssessment):
    """Muestra una evaluación de riesgos formateada en la interfaz."""
    with st.chat_message("assistant", avatar=f"data:image/png;base64,{get_base64_of_bin_file('assets/logo.png')}"):
        st.markdown(f"## Evaluación de Riesgos: {risk_data.sector}")
        
        # Mostrar marco regulatorio si existe
        if hasattr(risk_data, 'regulatory_framework') and risk_data.regulatory_framework:
            st.markdown(f"**Marco regulatorio:** {', '.join(risk_data.regulatory_framework)}")
            
        st.markdown(f"**Nivel de riesgo general:** {risk_data.overall_risk_level}")
        
        # Mostrar áreas impactadas
        st.markdown("### Áreas Impactadas")
        for area in risk_data.impacted_areas:
            risk_class = ""
            if area.impact_level == "Bajo":
                risk_class = "risk-low"
            elif area.impact_level == "Medio":
                risk_class = "risk-medium"
            elif area.impact_level == "Alto":
                risk_class = "risk-high"
            elif area.impact_level == "Crítico":
                risk_class = "risk-critical"
            
            st.markdown(f"""
            <div class="risk-area {risk_class}">
                <h4>{area.name}</h4>
                <p><strong>Impacto:</strong> {area.impact_level} | <strong>Probabilidad:</strong> {area.probability}</p>
                <p>{area.description}</p>
                <p><strong>Vulnerabilidades principales:</strong></p>
                <ul>
                    {"".join([f"<li>{vuln}</li>" for vuln in area.key_vulnerabilities])}
                </ul>
                <p><strong>Acciones de mitigación:</strong></p>
                <ul>
                    {"".join([f"<li>{action}</li>" for action in area.mitigation_actions])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar recomendaciones clave
        if risk_data.key_recommendations:
            st.markdown("### Recomendaciones Clave")
            for i, rec in enumerate(risk_data.key_recommendations, 1):
                st.markdown(f"{i}. {rec}")

def clear_old_messages():
    """Mantiene sólo los últimos mensajes para evitar acumulaciones excesivas."""
    max_messages = 20
    if len(st.session_state.messages) > max_messages * 2:
        st.session_state.messages = st.session_state.messages[-max_messages:]
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()

async def process_user_query(user_input: str):
    """Procesa la consulta del usuario a través del orquestador."""
    clear_old_messages()
    
    deps = OrchestratorDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    
    with st.spinner("Procesando tu consulta..."):
        # Mostrar un indicador de procesamiento mientras se analiza la consulta
        result = await process_query(user_input, deps)
        
        # Crear una respuesta adaptada al tipo de agente utilizado
        if result.agent_used == AgentType.RISK_ASSESSMENT:
            # Para mostrar evaluaciones de riesgo con formato especial
            display_risk_assessment(result.response)
            # Almacenar la respuesta como texto para el historial
            response_text = f"Evaluación de Riesgos para el sector: {result.response.sector}"
        else:
            # Para respuestas de compliance o informes
            st.markdown(result.response)
            response_text = result.response
        
        # Guardar los mensajes en el historial
        user_message = ModelRequest(parts=[UserPromptPart(content=user_input)])
        assistant_message = ModelResponse(parts=[TextPart(content=response_text)])
        
        st.session_state.messages.extend([user_message, assistant_message])
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()

def set_custom_style():
    st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5; }
        .sidebar .sidebar-content { background-color: #ffffff; }
        .new-chat-button {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #8B0D18;
            background-color: transparent;
            border: none;
            padding: 12px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.2s ease;
            width: 100%;
        }
        .new-chat-button:hover {
            background-color: rgba(139, 13, 24, 0.1);
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-header">
                <img src="data:image/png;base64,{get_base64_of_bin_file("assets/minsait_logo.png")}" class="minsait-logo">
            </div>
            """,
            unsafe_allow_html=True
        )
        # Botón para iniciar un nuevo chat (sin JS; se usa el botón de Streamlit)
        if st.button("Start new chat"):
            create_new_chat()
        st.markdown('Chats Recientes', unsafe_allow_html=True)
        recent_chats = list(st.session_state.get('chat_history', {}).items())
        if not recent_chats:
            st.markdown('_No hay chats recientes_')
        else:
            for chat_id, chat_data in recent_chats:
                chat_title = get_chat_title(chat_data)
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True):
                        switch_chat(chat_id)
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Eliminar chat"):
                        delete_chat(chat_id)

async def main():
    set_custom_style()
    # Mostrar logo en la parte superior cuando el sidebar esté colapsado
    st.markdown(
        f"""
        
        """,
        unsafe_allow_html=True
    )
    initialize_session_state()
    sidebar()
    st.title("AgentIA")
    
    # Mostrar los mensajes de la conversación actual
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    
    # Entrada de chat del usuario
    user_input = st.chat_input("¿Qué quieres saber?")
    if user_input:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Procesar la consulta a través del orquestador
        await process_user_query(user_input)

if __name__ == "__main__":
    asyncio.run(main())