import React, { useEffect, useState, useCallback } from 'react';
import { 
  Box, 
  Button, 
  Container, 
  Divider, 
  Grid, 
  Paper, 
  Tab, 
  Tabs, 
  TextField, 
  Typography,
  Snackbar,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  IconButton
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import ArticleIcon from '@mui/icons-material/Article';
import FeedbackIcon from '@mui/icons-material/Feedback';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { useQueryStore, useReportStore } from '../contexts/store';
import RichTextEditor from '../components/RichTextEditor';
import { DocxViewerFromUrl } from '../components/DocxViewer';
import { queryService, reportService, createWebSocketConnection } from '../services/api';

// Componente para mostrar el estado del reporte
const ReportStatus = ({ status }) => {
  let statusText = '';
  let statusColor = 'info';

  switch (status) {
    case 'generating':
      statusText = 'Generando reporte...';
      statusColor = 'info';
      break;
    case 'ready':
      statusText = 'Reporte listo';
      statusColor = 'success';
      break;
    case 'error':
      statusText = 'Error al generar el reporte';
      statusColor = 'error';
      break;
    default:
      statusText = 'Esperando consulta';
      statusColor = 'default';
  }

  return (
    <Alert severity={statusColor} variant="outlined" sx={{ my: 2 }}>
      {statusText}
      {status === 'generating' && (
        <CircularProgress size={20} sx={{ ml: 2 }} />
      )}
    </Alert>
  );
};

// Componente principal
const MainPage = () => {
  // Estado para la UI
  const [tabValue, setTabValue] = useState(0);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  
  // Estado global
  const { 
    query, setQuery, isLoading, response, 
    setResponse, startLoading, stopLoading, setError 
  } = useQueryStore();
  
  const {
    reportData, reportHtml, reportPath, status,
    setReportData, setReportHtml, setReportPath, setStatus,
    annotations, addAnnotation, selectedText, setSelectedText
  } = useReportStore();

  // Estado para WebSocket
  const [wsConnection, setWsConnection] = useState(null);

  // Función para manejar cambios en los tabs
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Función para enviar consulta
  const handleSubmitQuery = async () => {
    if (!query.trim()) {
      setNotification({
        open: true,
        message: 'Por favor, ingrese una consulta',
        severity: 'warning'
      });
      return;
    }

    try {
      startLoading();
      const data = await queryService.submitQuery(query);
      setResponse(data);
      
      // Automáticamente cambiamos a la pestaña de resultados
      setTabValue(1);
      
      setNotification({
        open: true,
        message: 'Consulta procesada correctamente',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error al procesar la consulta:', error);
      setError(error.message || 'Error al procesar la consulta');
      
      setNotification({
        open: true,
        message: 'Error al procesar la consulta: ' + (error.message || 'Desconocido'),
        severity: 'error'
      });
    } finally {
      stopLoading();
    }
  };

  // Función para generar reporte
  const handleGenerateReport = async () => {
    if (!response) {
      setNotification({
        open: true,
        message: 'Primero debe realizar una consulta',
        severity: 'warning'
      });
      return;
    }

    try {
      setStatus('generating');
      const data = await reportService.generateReport(query);
      setReportData(data);
      
      // Automáticamente cambiamos a la pestaña de reporte
      setTabValue(2);
      
      // Iniciar WebSocket para actualizaciones de estado
      initWebSocket(data.reportId);
      
      setNotification({
        open: true,
        message: 'Generación de reporte iniciada',
        severity: 'info'
      });
    } catch (error) {
      console.error('Error al iniciar generación de reporte:', error);
      setStatus('error');
      
      setNotification({
        open: true,
        message: 'Error al generar el reporte: ' + (error.message || 'Desconocido'),
        severity: 'error'
      });
    }
  };

  // Función para descargar reporte
  const handleDownloadReport = async () => {
    if (!reportData || !reportData.reportId) {
      setNotification({
        open: true,
        message: 'No hay reporte disponible para descargar',
        severity: 'warning'
      });
      return;
    }

    try {
      await reportService.downloadReport(reportData.reportId);
      
      setNotification({
        open: true,
        message: 'Descarga iniciada',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error al descargar el reporte:', error);
      
      setNotification({
        open: true,
        message: 'Error al descargar el reporte: ' + (error.message || 'Desconocido'),
        severity: 'error'
      });
    }
  };

  // Función para manejar anotaciones
  const handleAnnotationSubmit = (annotationText) => {
    if (!selectedText || !annotationText) return;
    
    const newAnnotation = {
      id: Date.now().toString(),
      selectedText,
      annotationText,
      timestamp: new Date().toISOString(),
    };
    
    addAnnotation(newAnnotation);
    setSelectedText('');
    
    setNotification({
      open: true,
      message: 'Anotación guardada',
      severity: 'success'
    });
  };

  // Función para cerrar notificaciones
  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  // Inicializar WebSocket
  const initWebSocket = useCallback((reportId) => {
    // Cerrar conexión anterior si existe
    if (wsConnection) {
      wsConnection.close();
    }

    const newConnection = createWebSocketConnection();
    
    newConnection.onConnect(() => {
      console.log('WebSocket conectado');
      // Enviar ID del reporte para suscribirse a actualizaciones
      newConnection.socket.send(JSON.stringify({ 
        action: 'subscribe', 
        reportId 
      }));
    });
    
    newConnection.onMessage((data) => {
      console.log('WebSocket mensaje recibido:', data);
      
      if (data.status) {
        setStatus(data.status);
      }
      
      if (data.reportPath) {
        setReportPath(data.reportPath);
      }
      
      if (data.reportHtml) {
        setReportHtml(data.reportHtml);
      }
      
      if (data.status === 'ready') {
        setNotification({
          open: true,
          message: 'Reporte generado correctamente',
          severity: 'success'
        });
      }
    });
    
    newConnection.onError((error) => {
      console.error('WebSocket error:', error);
      setNotification({
        open: true,
        message: 'Error en la conexión en tiempo real',
        severity: 'error'
      });
    });
    
    setWsConnection(newConnection);
    
    // Cleanup function
    return () => {
      if (newConnection) {
        newConnection.close();
      }
    };
  }, [wsConnection, setStatus, setReportPath, setReportHtml]);

  // Limpiar WebSocket al desmontar
  useEffect(() => {
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, [wsConnection]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Experto en Normativas VISA y Mastercard
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Herramienta para consultas y generación de reportes sobre normativas de VISA y Mastercard.
        </Typography>
      </Paper>

      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
        sx={{ mb: 3 }}
      >
        <Tab label="Consulta" icon={<SendIcon />} iconPosition="start" />
        <Tab label="Resultados" icon={<ArticleIcon />} iconPosition="start" disabled={!response} />
        <Tab label="Reporte" icon={<VisibilityIcon />} iconPosition="start" disabled={!reportData} />
        <Tab label="Anotaciones" icon={<FeedbackIcon />} iconPosition="start" disabled={!reportData} />
      </Tabs>

      {/* Pestaña de Consulta */}
      {tabValue === 0 && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Ingrese su consulta sobre normativas VISA y Mastercard
          </Typography>
          
          <TextField
            label="Consulta"
            placeholder="Ej: ¿Cuáles son los requisitos para cumplir con PCI DSS en comercios electrónicos?"
            multiline
            rows={4}
            fullWidth
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            variant="outlined"
            margin="normal"
          />
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              variant="contained"
              color="primary"
              endIcon={<SendIcon />}
              onClick={handleSubmitQuery}
              disabled={isLoading || !query.trim()}
            >
              {isLoading ? 'Procesando...' : 'Enviar Consulta'}
              {isLoading && <CircularProgress size={20} sx={{ ml: 1 }} />}
            </Button>
          </Box>
        </Paper>
      )}

      {/* Pestaña de Resultados */}
      {tabValue === 1 && response && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Resultados de la consulta
          </Typography>
          
          <Paper variant="outlined" sx={{ p: 2, mb: 3, bgcolor: '#f9f9f9' }}>
            <Typography variant="subtitle2" color="text.secondary">
              Consulta:
            </Typography>
            <Typography variant="body1" paragraph>
              {query}
            </Typography>
          </Paper>
          
          <Typography variant="subtitle2" color="text.secondary">
            Respuesta:
          </Typography>
          
          <Box sx={{ 
            my: 2, 
            p: 2, 
            border: '1px solid #e0e0e0', 
            borderRadius: 1,
            backgroundColor: '#fff',
            whiteSpace: 'pre-line'
          }}>
            {response.response || response}
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
            <Button
              variant="contained"
              color="secondary"
              endIcon={<ArticleIcon />}
              onClick={handleGenerateReport}
              disabled={status === 'generating'}
            >
              Generar Reporte
            </Button>
          </Box>
        </Paper>
      )}

      {/* Pestaña de Reporte */}
      {tabValue === 2 && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Reporte Normativo
          </Typography>
          
          <ReportStatus status={status} />
          
          {reportPath && status === 'ready' && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Previsualización del reporte:
              </Typography>
              
              <DocxViewerFromUrl 
                url={`/api/report/preview/${reportData.reportId}`} 
                viewerType="mammoth" 
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownloadReport}
                >
                  Descargar Reporte
                </Button>
              </Box>
            </Box>
          )}
          
          {status === 'generating' && (
            <Box display="flex" justifyContent="center" alignItems="center" my={4} py={4}>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ ml: 2 }}>
                Generando reporte...
              </Typography>
            </Box>
          )}
          
          {status === 'error' && (
            <Alert severity="error" sx={{ my: 3 }}>
              Ocurrió un error al generar el reporte. Por favor, intente nuevamente.
            </Alert>
          )}
        </Paper>
      )}

      {/* Pestaña de Anotaciones */}
      {tabValue === 3 && reportData && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Anotaciones y Comentarios
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              {status === 'ready' && reportHtml && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Reporte (seleccione texto para añadir anotaciones):
                  </Typography>
                  
                  <Box
                    sx={{
                      border: '1px solid #ddd',
                      borderRadius: 1,
                      p: 2,
                      mb: 2,
                      maxHeight: '500px',
                      overflowY: 'auto'
                    }}
                    dangerouslySetInnerHTML={{ __html: reportHtml }}
                    onMouseUp={() => {
                      const selection = window.getSelection();
                      if (selection && selection.toString()) {
                        setSelectedText(selection.toString());
                      }
                    }}
                  />
                </Box>
              )}
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Añadir anotación:
                </Typography>
                
                {selectedText && (
                  <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: '#f5f5f5' }}>
                    <Typography variant="body2" color="text.secondary">
                      Texto seleccionado:
                    </Typography>
                    <Typography variant="body1">
                      "{selectedText}"
                    </Typography>
                  </Paper>
                )}
                
                <RichTextEditor
                  content=""
                  onChange={(html) => {
                    if (selectedText && html) {
                      handleAnnotationSubmit(html);
                    }
                  }}
                  placeholder="Escriba su anotación aquí..."
                />
              </Box>
              
              <Divider sx={{ my: 3 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Anotaciones guardadas:
              </Typography>
              
              {annotations.length === 0 ? (
                <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    No hay anotaciones guardadas.
                  </Typography>
                </Paper>
              ) : (
                <Box sx={{ maxHeight: '400px', overflowY: 'auto' }}>
                  {annotations.map((annotation) => (
                    <Card key={annotation.id} variant="outlined" sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Texto anotado:
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 2, fontStyle: 'italic' }}>
                          "{annotation.selectedText}"
                        </Typography>
                        
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Anotación:
                        </Typography>
                        <Box dangerouslySetInnerHTML={{ __html: annotation.annotationText }} />
                        
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                          {new Date(annotation.timestamp).toLocaleString()}
                        </Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}
            </Grid>
          </Grid>
        </Paper>
      )}

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
    </Container>
  );
};

export default MainPage;