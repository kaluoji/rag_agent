import React, { useState } from 'react';
import { 
  Box, 
  Container,
  Snackbar,
  Alert,
  Paper,
  Typography
} from '@mui/material';
import { useQueryStore } from '../contexts/store';
import { queryService } from '../services/api';
import ChatInterface from '../components/ChatInterface';

// Componente principal
const MainPage = () => {
  // Estado para la UI
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  
  // Estado global
  const { 
    setQuery, isLoading, 
    setResponse, startLoading, stopLoading, setError 
  } = useQueryStore();

  // Función para enviar consulta
  const handleSubmitQuery = async (queryText, addResponseCallback, documents = null) => {
    if (!queryText.trim()) {
      setNotification({
        open: true,
        message: 'Por favor, ingrese una consulta',
        severity: 'warning'
      });
      return;
    }

    try {
      // Actualizar estado global
      setQuery(queryText);
      startLoading();
      
      // Enviar consulta al servicio
      const data = await queryService.submitQueryWithDocuments(queryText, documents);
      
      // Actualizar estado global con la respuesta
      setResponse(data);
      
      // Agregar respuesta al chat
      if (addResponseCallback) {
        const responseText = data.response || JSON.stringify(data);
        addResponseCallback(responseText);
      }
      
      // Mensaje de éxito diferente si hay documentos
      const successMessage = documents && documents.length > 0 ? 
        'Análisis GAP completado correctamente' : 
        'Consulta procesada correctamente';

      setNotification({
        open: true,
        message: successMessage,
        severity: 'success'
      });
    } catch (error) {
      console.error('Error al procesar la consulta:', error);
      setError(error.message || 'Error al procesar la consulta');
      
      // Agregar mensaje de error al chat
      if (addResponseCallback) {
        addResponseCallback(`Error al procesar la consulta: ${error.message || 'Desconocido'}`);
      }
      
      setNotification({
        open: true,
        message: 'Error al procesar la consulta: ' + (error.message || 'Desconocido'),
        severity: 'error'
      });
    } finally {
      stopLoading();
    }
  };

  // Función para cerrar notificaciones
  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  return (
    <Box 
      className="chat-page-container"
      sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        height: 'calc(100vh - 64px)', 
        overflow: 'hidden', 
        bgcolor: '#f5f7fa' 
      }}
    >
      <Container 
        maxWidth="lg" 
        sx={{ 
          flexGrow: 1, 
          py: 4, 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden' // Importante para contener el scroll
        }}
      >
        <Paper 
          elevation={0} 
          sx={{ 
            flexGrow: 1, 
            display: 'flex', 
            flexDirection: 'column',
            overflow: 'hidden', // Importante para que el scroll funcione dentro
            borderRadius: '24px',
            boxShadow: '0 2px 12px rgba(0, 0, 0, 0.08)',
            border: '1px solid rgba(0, 0, 0, 0.12)'
          }}
        >
          <ChatInterface 
            onSubmitQuery={handleSubmitQuery}
            isLoading={isLoading}
          />
        </Paper>
      </Container>
      
      {/* Notificaciones */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default MainPage;