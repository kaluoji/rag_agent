from __future__ import annotations
from typing import Literal
import asyncio
import json
import os
import sys
import PyPDF2

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

# Configuración de la página
st.set_page_config(
    page_title="AgentIA",
    layout="wide"
)

# Estilos CSS

st.markdown("""
<style>
    /* Estilos existentes ... */
    
    /* Estilos para el informe de riesgos */
    .risk-area {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .risk-area h4 {
        margin-top: 0;
        margin-bottom: 15px;
        color: #333;
        font-size: 1.2em;
    }
    
    .risk-area ul {
        margin-bottom: 15px;
        padding-left: 20px;
    }
    
    .risk-area li {
        margin-bottom: 10px;
        line-height: 1.5;
    }
    
    .risk-low { border-left: 4px solid #28a745; }
    .risk-medium { border-left: 4px solid #ffc107; }
    .risk-high { border-left: 4px solid #fd7e14; }
    .risk-critical { border-left: 4px solid #dc3545; }
    
    .risk-area strong {
        color: #444;
    }
    
    .risk-area em {
        color: #666;
        font-style: italic;
    }

    .citation {
    margin: 15px 0;
    padding: 12px 15px;
    border-left: 4px solid #8B0D18;
    background-color: #ffffff;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .citation em {
        font-style: italic;
        color: #555;
    }

    .blockquote-citation {
        margin: 20px 0;
        padding: 15px;
        background-color: #ffffff;
        border-left: 4px solid #8B0D18;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-style: italic;
        color: #555;
    }
    
    .new-chat-button {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #8B0D18;
        background-color: transparent;
        border: none;
        padding: 20px;
        margin-top: 2px;
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
            # Procesamos el contenido para formatear correctamente las citas
            content = process_citations(part.content)
            st.markdown(content, unsafe_allow_html=True)

def process_citations(content):
    """
    Procesa el contenido para dar formato adecuado a las citas.
    Convierte etiquetas HTML específicas y formatea las citas correctamente.
    """
    import re
    
    # Reemplazar <strong>Cita:</strong> <em>"texto"</em> con un formato especial
    content = re.sub(
        r'<strong>Cita:</strong>\s*<em>"(.+?)"</em>',
        r'<div class="citation"><strong>Cita:</strong> <em>"\1"</em></div>',
        content
    )
    
    # Reemplazar <blockquote><em>"texto"</em></blockquote> con un formato especial
    content = re.sub(
        r'<blockquote><em>"(.+?)"</em></blockquote>',
        r'<div class="blockquote-citation"><em>"\1"</em></div>',
        content
    )
    
    return content

def clear_old_messages():
    """Mantiene sólo los últimos mensajes para evitar acumulaciones excesivas."""
    max_messages = 20
    if len(st.session_state.messages) > max_messages * 2:
        st.session_state.messages = st.session_state.messages[-max_messages:]
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae el texto de un archivo PDF y lo retorna."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        return f"Error al extraer texto del PDF: {str(e)}"

async def process_user_query(user_input: str):
    """Procesa la consulta del usuario a través del orquestador e integra el contenido de documentos adjuntos."""
    clear_old_messages()
    
    deps = OrchestratorDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    
    # Integrar documentos adjuntos en la consulta
    documents_context = ""
    if "documents" in st.session_state and st.session_state["documents"]:
        documents_context += "\n\n### Documentos Adjuntos:\n"
        for doc in st.session_state["documents"]:
            try:
                # Si el documento es un PDF, extrae el texto; de lo contrario, intenta leerlo como texto
                if doc["name"].lower().endswith(".pdf") or doc["type"] == "application/pdf":
                    content = extract_text_from_pdf(doc["path"])
                else:
                    with open(doc["path"], "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                
                # Limitar el contenido a 1000 caracteres para no exceder límites de tokens
                max_chars = 1000
                if len(content) > max_chars:
                    content = content[:max_chars] + "...\n"
                documents_context += f"\n**{doc['name']}**:\n{content}\n"
            except Exception as e:
                documents_context += f"\n**{doc['name']}**: Error al leer el contenido: {str(e)}\n"
    
    # Combinar la consulta original con el contexto extraído de los documentos
    final_query = user_input + documents_context

    with st.spinner("Procesando tu consulta..."):
        result = await process_query(final_query, deps)
        
        # Crear mensaje de usuario para mostrar
        user_message = ModelRequest(parts=[UserPromptPart(content=user_input)])
        
        # Extraer y formatear el contenido de la respuesta
        try:
            # Verificar cómo viene la respuesta
            if isinstance(result.response, str):
                response_text = result.response
            elif hasattr(result.response, 'data'):
                response_text = result.response.data
            else:
                response_text = str(result.response)
                
            # Crear mensaje de respuesta del asistente
            assistant_message = ModelResponse(parts=[TextPart(content=response_text)])
            
            # Añadir a la lista de mensajes
            st.session_state.messages.extend([user_message, assistant_message])
            
            # Guardar en el historial
            if st.session_state.current_chat_id:
                st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()
            
            # Mostrar la respuesta
            with st.chat_message("assistant", avatar=f"data:image/png;base64,{get_base64_of_bin_file('assets/logo.png')}"):
                content = process_citations(response_text)
                st.markdown(content, unsafe_allow_html=True)
                
        except Exception as e:
            # Manejar errores en la visualización de la respuesta
            st.error(f"Error al mostrar la respuesta: {str(e)}")
            # Intentar mostrar la respuesta en bruto
            st.write("Respuesta en bruto del agente:", result.response)


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
            padding: 20px;
            margin-top: 2px;
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

def handle_document_upload():
    """
    Maneja la carga de documentos y los procesa para su uso en el sistema.
    Devuelve una lista de documentos procesados si se cargan correctamente.
    """
    st.sidebar.markdown("## Adjuntar Documentos")
    
    uploaded_files = st.sidebar.file_uploader(
        "Sube documentos relevantes para el análisis",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "xlsx"]
    )
    
    if not uploaded_files:
        return None
    
    documents = []
    
    with st.sidebar.expander("Documentos cargados", expanded=True):
        for uploaded_file in uploaded_files:
            # Mostrar información del archivo
            file_details = {
                "Nombre": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamaño": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.write(f"**{file_details['Nombre']}**")
            st.write(f"Tipo: {file_details['Tipo']} | Tamaño: {file_details['Tamaño']}")
            
            # Guardar temporalmente el archivo
            try:
                temp_file_path = os.path.join("temp_files", uploaded_file.name)
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                doc_info = {
                    "name": uploaded_file.name,
                    "path": temp_file_path,
                    "type": uploaded_file.type
                }
                
                documents.append(doc_info)
                st.success(f"✅ Documento guardado: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error al procesar {uploaded_file.name}: {str(e)}")
    
    return documents if documents else None



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
      # Integración del manejo de carga de documentos
    documents = handle_document_upload()
    if documents:
        st.session_state["documents"] = documents

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