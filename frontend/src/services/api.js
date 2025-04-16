import axios from 'axios';

const API_URL = 'http://localhost:8000';  // URL base del backend

// Configuración base para axios
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para manejar errores globalmente
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response || error);
    return Promise.reject(error);
  }
);

// Servicio de consultas normativas
export const queryService = {
  // Enviar una consulta al sistema multi-agente
  submitQuery: async (queryText) => {
    const response = await api.post('/api/query', { query: queryText });  // Note the /api/ prefix
    return response.data;
  },
};

// Servicio de generación y gestión de reportes
export const reportService = {
  // Generar un nuevo reporte basado en una consulta
  generateReport: async (queryText) => {
    const response = await api.post('/report/generate', { query: queryText });
    return response.data;
  },

  // Obtener reporte en formato HTML para previsualización
  getReportPreview: async (reportId) => {
    const response = await api.get(`/report/preview/${reportId}`);
    return response.data;
  },

  // Obtener información sobre el reporte
  getReportInfo: async (reportId) => {
    const response = await api.get(`/report/${reportId}`);
    return response.data;
  },

  // Guardar anotaciones o ediciones en un reporte
  saveAnnotations: async (reportId, annotations) => {
    const response = await api.post(`/report/${reportId}/annotations`, { annotations });
    return response.data;
  },

  // Descargar reporte en formato Word
  downloadReport: async (reportId) => {
    const response = await api.get(`/report/${reportId}/download`, {
      responseType: 'blob',
    });
    
    // Crear un objeto URL para el blob y activar la descarga
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `reporte-${reportId}.docx`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    
    return true;
  },
};

// Servicio WebSocket para actualizaciones en tiempo real
export const createWebSocketConnection = () => {
  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsHost = window.location.host; // Incluye el dominio y el puerto
  const wsURL = `${wsProtocol}//localhost:8000/ws/report`; 
  
  const socket = new WebSocket(wsURL);
  
  return {
    socket,
    
    // Establece un manejador para eventos recibidos
    onMessage: (callback) => {
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          callback(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    },
    
    // Establece un manejador para errores
    onError: (callback) => {
      socket.onerror = (error) => {
        callback(error);
      };
    },
    
    // Establece un manejador para la conexión
    onConnect: (callback) => {
      socket.onopen = () => {
        callback();
      };
    },
    
    // Establece un manejador para la desconexión
    onDisconnect: (callback) => {
      socket.onclose = () => {
        callback();
      };
    },
    
    // Cierra la conexión WebSocket
    close: () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    },
  };
};