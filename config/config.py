# config/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_model: str = 'gpt-4o-mini'
    openai_api_key: str
    supabase_url: str
    supabase_service_key: str

    class Config:
        env_file = ".env"

settings = Settings()
