# config/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #llm_model: str = 'ft:gpt-4o-mini-2024-07-18:personal::BCAkAcVL'
    llm_model: str = "gpt-4o-mini"  
    tokenizer_model: str = "gpt-4o-mini"  # Modelo para conteo de tokens compatible con tiktoken
    openai_api_key: str
    supabase_url: str
    supabase_key: str
    supabase_service_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
