// frontend/src/services/api.js
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Configuración base para axios
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para procesar respuestas y dar formato consistente
api.interceptors.response.use(
  (response) => {
    // Procesar tipos de respuesta comunes y normalizarlos
    if (response.data) {
      // Si la respuesta es un objeto con la propiedad response, es el formato esperado
      if (typeof response.data === 'object' && response.data.response) {
        return response.data;
      }
      
      // Si es un string, envolverlo en un objeto con estructura consistente
      if (typeof response.data === 'string') {
        return {
          response: response.data,
          metadata: {}
        };
      }
      
      // Por defecto, devolver los datos tal cual
      return response.data;
    }
    
    return response;
  },
  (error) => {
    console.error('API Error:', error.response || error);
    return Promise.reject(error);
  }
);

// Servicio de consultas normativas
export const queryService = {
  // Enviar una consulta al sistema multi-agente
  submitQuery: async (queryText) => {
    try {
      const response = await api.post('/api/query', { query: queryText });
      return response;
    } catch (error) {
      console.error('Error en la consulta:', error);
      throw error;
    }
  },
};

// Servicio de generación y gestión de reportes
export const reportService = {
  // Generar un nuevo reporte basado en una consulta
  generateReport: async (queryText) => {
    try {
      const response = await api.post('/api/report/generate', { 
        query: queryText,
        format: 'docx'  // Formato por defecto
      });
      return response;
    } catch (error) {
      console.error('Error al generar reporte:', error);
      throw error;
    }
  },

  // Obtener reporte en formato HTML para previsualización
  getReportPreview: async (reportId) => {
    try {
      const response = await api.get(`/api/report/preview/${reportId}`);
      return response;
    } catch (error) {
      console.error('Error al obtener vista previa:', error);
      throw error;
    }
  },

  // Obtener información sobre el reporte
  getReportInfo: async (reportId) => {
    try {
      const response = await api.get(`/api/report/${reportId}`);
      return response;
    } catch (error) {
      console.error('Error al obtener información del reporte:', error);
      throw error;
    }
  },

  // Guardar anotaciones o ediciones en un reporte
  saveAnnotations: async (reportId, annotations) => {
    try {
      const response = await api.post(`/api/report/${reportId}/annotations`, { annotations });
      return response;
    } catch (error) {
      console.error('Error al guardar anotaciones:', error);
      throw error;
    }
  },

  // Descargar reporte en formato Word
  downloadReport: async (reportId) => {
    try {
      const response = await api.get(`/api/report/${reportId}/download`, {
        responseType: 'blob',
      });
      
      // Crear un objeto URL para el blob y activar la descarga
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `reporte-normativo-${reportId}.docx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      return true;
    } catch (error) {
      console.error('Error al descargar el reporte:', error);
      throw error;
    }
  },
};

// Servicio WebSocket para actualizaciones en tiempo real
export const createWebSocketConnection = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = import.meta.env.VITE_API_WS_HOST || window.location.host;
  const wsURL = `${protocol}//${host}/ws/report`; 
  
  try {
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
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.close();
        }
      },
    };
  } catch (error) {
    console.error('Error al crear conexión WebSocket:', error);
    // Devolver un objeto con métodos vacíos para evitar errores
    return {
      socket: null,
      onMessage: () => {},
      onError: () => {},
      onConnect: () => {},
      onDisconnect: () => {},
      close: () => {},
    };
  }
};

