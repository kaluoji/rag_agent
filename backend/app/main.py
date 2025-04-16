# app/main.py
import os
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError

# Obtener la ruta del directorio donde está el archivo main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

# Agregar el directorio raíz al path para que las importaciones funcionen correctamente
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Intentar cargar .env desde varias ubicaciones posibles
dotenv_paths = [
    os.path.join(root_dir, ".env"),  # backend/.env
    os.path.join(os.path.dirname(root_dir), ".env"),  # ../.env (directorio padre)
]

for dotenv_path in dotenv_paths:
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Cargado archivo .env desde: {dotenv_path}")
        break
else:
    print("ADVERTENCIA: No se encontró ningún archivo .env")

from app.core.config import settings
from app.core.websocket import ConnectionManager
from app.api.routes import api_router

import logging


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # URL del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejador global de excepciones
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_traceback = traceback.format_exc()
    logger.error(f"Error en la ruta {request.url.path}: {str(exc)}")
    logger.error(error_traceback)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "detail": error_traceback,
            "path": request.url.path
        }
    )

# Manejador para errores de validación
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Error de validación",
            "detail": exc.errors(),
            "path": request.url.path
        }
    )

# Incluir rutas API
app.include_router(api_router, prefix="/api")

# Crear directorio de reportes si no existe
os.makedirs(settings.REPORTS_DIR, exist_ok=True)

# Servir carpeta de reportes como archivos estáticos
app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# Gestor de conexiones WebSocket
connection_manager = ConnectionManager()

@app.websocket("/ws/report")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Si el mensaje contiene una acción de suscripción a un reporte específico
            if data.get("action") == "subscribe" and data.get("reportId"):
                report_id = data.get("reportId")
                logger.info(f"Cliente suscrito al reporte: {report_id}")
                connection_manager.add_to_group(websocket, f"report_{report_id}")
                
                # Enviar confirmación de suscripción
                await websocket.send_json({
                    "event": "subscribed",
                    "reportId": report_id
                })
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {str(e)}")
        connection_manager.disconnect(websocket)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API del Experto en Normativas VISA y Mastercard"}

# Endpoint de prueba para verificar la conectividad
@app.get("/api/test")
def test_endpoint():
    return {"message": "Conexión exitosa con el backend"}

# Fallback para servir el index.html de React para rutas no encontradas (SPA)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Si es una ruta de API, devolver 404
    if full_path.startswith("api/"):
        return {"detail": "Not Found"}
    
    # Para cualquier otra ruta, servir el index.html de la SPA
    frontend_path = os.path.join(settings.STATIC_DIR, "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return {"message": "Frontend not built. Please run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)