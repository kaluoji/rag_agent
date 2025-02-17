# ui/session_manager.py
import streamlit as st
from datetime import datetime
from pydantic_ai.messages import ModelRequest

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chat_history[chat_id] = []
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    st.experimental_rerun()

def switch_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = st.session_state.chat_history[chat_id]
        st.experimental_rerun()

def get_chat_title(messages) -> str:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if getattr(part, "part_kind", "") == "user-prompt":
                    return part.content[:30] + "..." if len(part.content) > 30 else part.content
    return "Nueva conversación"

def delete_chat(chat_id):
    if chat_id in st.session_state.chat_history:
        del st.session_state.chat_history[chat_id]
        if chat_id == st.session_state.current_chat_id:
            st.session_state.current_chat_id = None
            st.session_state.messages = []
        st.experimental_rerun()
