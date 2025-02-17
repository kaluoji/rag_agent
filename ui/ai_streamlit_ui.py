# ui/ai_streamlit_ui.py

import asyncio
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
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from agents.ai_expert_v1 import ai_expert, AIDeps
from ui.session_manager import initialize_session_state, create_new_chat, switch_chat, get_chat_title, delete_chat

st.set_page_config(page_title="AgentIA", layout="wide")

# Incluir tus estilos CSS personalizados
st.markdown("""
<style>
    /* Tus estilos CSS aquí */
    blockquote {
        font-style: italic;
        color: #555;
        border-left: 4px solid #ccc;
        margin: 1em 0;
        padding-left: 1em;
    }
    em {
        font-style: italic;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = Client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
logfire.configure(send_to_logfire='never')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_message_part(part):
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}", unsafe_allow_html=True)
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content, unsafe_allow_html=True)
    elif part.part_kind == 'text':
        with st.chat_message("assistant", avatar=f"data:image/png;base64,{get_base64_of_bin_file('assets/logo.png')}"):
            st.markdown(part.content, unsafe_allow_html=True)

def clear_old_messages():
    max_messages = 20
    if len(st.session_state.messages) > max_messages * 2:
        st.session_state.messages = st.session_state.messages[-max_messages:]
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()

async def run_agent_with_streaming(user_input: str):
    clear_old_messages()
    deps = AIDeps(supabase=supabase, openai_client=openai_client)
    max_messages = 20
    message_history = st.session_state.messages[-max_messages:] if len(st.session_state.messages) > max_messages else st.session_state.messages
    async with ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=message_history,
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)
        filtered_messages = [msg for msg in result.new_messages() 
                             if not (hasattr(msg, 'parts') and any(getattr(part, 'part_kind', "") == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(ModelResponse(parts=[TextPart(content=partial_text)]))
        if st.session_state.current_chat_id:
            st.session_state.chat_history[st.session_state.current_chat_id] = st.session_state.messages.copy()

def set_custom_style():
    st.markdown("""
    <style>
        .stApp { background-color: #f5f5f5; }
        .sidebar .sidebar-content { background-color: #ffffff; }
        .new-chat-button { /* estilos */ }
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
    initialize_session_state()
    sidebar()
    st.title("AgentIA")
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    user_input = st.chat_input("¿Qué quieres saber?")
    if user_input:
        st.session_state.messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())
