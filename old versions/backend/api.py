from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from ai_expert_v0 import ai_expert  # Importamos tu agente existente

app = FastAPI()

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Puerto por defecto de Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validar los datos
class ChatMessage(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Aquí utilizamos tu ai_expert existente
        response = await ai_expert.run(message.message)
        return ChatResponse(
            response=response,
            chat_id=message.chat_id or "new_chat"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Aquí puedes procesar los archivos subidos
        # Por ejemplo, guardarlos temporalmente y procesarlos con tu RAG
        file_names = [file.filename for file in files]
        return {"message": f"Archivos recibidos: {file_names}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats")
async def get_chats():
    try:
        # Aquí puedes implementar la lógica para obtener el historial de chats
        # Por ahora retornamos un ejemplo
        return {
            "chats": [
                {"id": "1", "name": "Chat 1"},
                {"id": "2", "name": "Chat 2"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Punto de entrada principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)