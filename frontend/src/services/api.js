// frontend/src/services/api.js (modificado)
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
  
  // Obtener el estado de una consulta
  getQueryStatus: async (queryId) => {
    try {
      const response = await api.get(`/api/query/${queryId}/status`);
      return response;
    } catch (error) {
      console.error('Error al obtener estado de consulta:', error);
      throw error;
    }
  },
};

// Servicio de gestión de reportes
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

  // Guardar anotaciones para un documento
  saveAnnotations: async (reportId, annotations) => {
    try {
      const response = await api.post(`/api/report/annotations/${reportId}`, {
        annotations
      });
      return response;
    } catch (error) {
      console.error('Error al guardar anotaciones:', error);
      throw error;
    }
  },

  // Obtener anotaciones de un documento
  getAnnotations: async (reportId) => {
    try {
      const response = await api.get(`/api/report/annotations/${reportId}`);
      return response.annotations || [];
    } catch (error) {
      console.error('Error al obtener anotaciones:', error);
      return [];
    }
  },

  // Obtener vista previa del documento en HTML
  getReportPreview: async (reportPath) => {
    try {
      const response = await api.get(`/api/report/preview`, {
        params: { path: reportPath }
      });
      return response;
    } catch (error) {
      console.error('Error al obtener vista previa:', error);
      throw error;
    }
  },
  
  // Obtener contenido del reporte por ID
  getReportContentById: async (reportId) => {
    try {
      const response = await api.get(`/api/report/content_by_id/${reportId}`);
      return response;
    } catch (error) {
      console.error('Error al obtener contenido del reporte por ID:', error);
      throw error;
    }
  },

  // Descargar reporte
  downloadReport: async (reportPath) => {
    try {
      const response = await api.get(`/api/report/download`, {
        params: { path: reportPath },
        responseType: 'blob',
      });
      
      // Extraer nombre del archivo de la ruta
      const filename = reportPath.split('/').pop() || 'reporte-normativo.docx';
      
      // Crear un objeto URL para el blob y activar la descarga
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
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

// Función para manejar errores del backend de forma transparente
const handleFallback = async (apiCall, fallbackFn) => {
  try {
    return await apiCall();
  } catch (error) {
    console.warn('API no disponible o error, usando fallback:', error);
    return fallbackFn();
  }
};

// Funciones de fallback que generan datos simulados
const fallbacks = {
  // Fallback para obtener el contenido de un reporte
  getReportContent: (reportPath) => {
    console.log('Usando fallback para getReportContent:', reportPath);
    // Solo simulamos una respuesta exitosa, el contenido real se genera en el componente
    return Promise.resolve({
      success: true,
      message: 'Contenido simulado generado localmente',
      filename: reportPath.split('/').pop() || 'documento.docx',
      base64Content: null // No proporcionamos contenido real, lo generará el componente
    });
  },
  
  // Fallback para descargar un reporte
  downloadReport: (reportPath) => {
    console.log('Usando fallback para downloadReport:', reportPath);
    // Mostrar mensaje al usuario
    alert('La descarga directa no está disponible. El backend aún no implementa esta funcionalidad.');
    return Promise.resolve(false);
  }
};

// Extender el servicio de reportes con métodos que usan fallback
export const reportServiceWithFallback = {
  // Obtener el contenido del documento como base64
  getReportContent: async (reportPath) => {
    return handleFallback(
      () => reportService.getReportContent(reportPath),
      () => fallbacks.getReportContent(reportPath)
    );
  },

  // Obtener contenido por ID con fallback
  getReportContentById: async (reportId) => {
    return handleFallback(
      () => reportService.getReportContentById(reportId),
      () => fallbacks.getReportContent(`Reporte_Normativo_${reportId}.docx`)
    );
  },

  // Descargar reporte con fallback
  downloadReport: async (reportPath) => {
    return handleFallback(
      () => reportService.downloadReport(reportPath),
      () => fallbacks.downloadReport(reportPath)
    );
  }
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