from fastapi import APIRouter
from app.api.query import router as query_router
from app.api.report import router as report_router

# Crear el router principal de la API
api_router = APIRouter()

# Incluir los routers de los diferentes módulos
api_router.include_router(query_router, prefix="/query", tags=["query"])
api_router.include_router(report_router, prefix="/report", tags=["report"])