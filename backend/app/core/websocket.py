from fastapi import WebSocket
from typing import Dict, List, Set, Any
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Gestiona las conexiones WebSocket y los grupos de conexiones.
    Permite enviar mensajes a clientes específicos o a grupos de clientes.
    """
    
    def __init__(self):
        # Todas las conexiones activas
        self.active_connections: List[WebSocket] = []
        # Conexiones agrupadas por categoría (ej: report_id, user_id)
        self.connection_groups: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """
        Acepta una nueva conexión WebSocket.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nueva conexión WebSocket establecida. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Cierra y elimina una conexión WebSocket.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Eliminar de todos los grupos donde esté presente
        for group_name in list(self.connection_groups.keys()):
            if websocket in self.connection_groups[group_name]:
                self.connection_groups[group_name].remove(websocket)
                # Si el grupo queda vacío, eliminarlo
                if not self.connection_groups[group_name]:
                    del self.connection_groups[group_name]
        
        logger.info(f"Conexión WebSocket cerrada. Restantes: {len(self.active_connections)}")
    
    def add_to_group(self, websocket: WebSocket, group_name: str):
        """
        Añade una conexión a un grupo específico.
        """
        if group_name not in self.connection_groups:
            self.connection_groups[group_name] = set()
        
        self.connection_groups[group_name].add(websocket)
        logger.info(f"Conexión añadida al grupo '{group_name}'. Total en grupo: {len(self.connection_groups[group_name])}")
    
    def remove_from_group(self, websocket: WebSocket, group_name: str):
        """
        Elimina una conexión de un grupo específico.
        """
        if group_name in self.connection_groups and websocket in self.connection_groups[group_name]:
            self.connection_groups[group_name].remove(websocket)
            logger.info(f"Conexión eliminada del grupo '{group_name}'")
            
            # Si el grupo queda vacío, eliminarlo
            if not self.connection_groups[group_name]:
                del self.connection_groups[group_name]
                logger.info(f"Grupo '{group_name}' eliminado por estar vacío")
    
    async def send_personal_message(self, message: Any, websocket: WebSocket):
        """
        Envía un mensaje a una conexión específica.
        """
        try:
            if isinstance(message, dict):
                await websocket.send_json(message)
            elif isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_json({"message": str(message)})
        except Exception as e:
            logger.error(f"Error al enviar mensaje personal: {str(e)}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Any):
        """
        Envía un mensaje a todas las conexiones activas.
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                if isinstance(message, dict):
                    await connection.send_json(message)
                elif isinstance(message, str):
                    await connection.send_text(message)
                else:
                    await connection.send_json({"message": str(message)})
            except Exception as e:
                logger.error(f"Error al enviar broadcast: {str(e)}")
                disconnected.append(connection)
        
        # Desconectar conexiones con errores
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_group(self, group_name: str, message: Any):
        """
        Envía un mensaje a todas las conexiones de un grupo específico.
        """
        if group_name not in self.connection_groups:
            logger.warning(f"Intento de broadcast al grupo '{group_name}' que no existe")
            return
        
        disconnected = []
        for connection in self.connection_groups[group_name]:
            try:
                if isinstance(message, dict):
                    await connection.send_json(message)
                elif isinstance(message, str):
                    await connection.send_text(message)
                else:
                    await connection.send_json({"message": str(message)})
            except Exception as e:
                logger.error(f"Error al enviar broadcast a grupo: {str(e)}")
                disconnected.append(connection)
        
        # Desconectar conexiones con errores
        for connection in disconnected:
            self.disconnect(connection)
        
        logger.info(f"Mensaje enviado al grupo '{group_name}'. Conexiones activas: {len(self.connection_groups[group_name])}")