// frontend/src/components/DocxPreviewPanel.jsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  IconButton, 
  Collapse,
  CircularProgress,
  Button,
  Divider,
  Tooltip,
  Alert
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';
import mammoth from 'mammoth';
import { styled } from '@mui/material/styles';

// Componente estilizado para el panel lateral
const PreviewPanel = styled(Paper)(({ theme }) => ({
  position: 'fixed',
  right: 0,
  top: 64, // Altura del AppBar
  bottom: 0,
  width: '40%', // Ancho del panel
  maxWidth: 600,
  zIndex: theme.zIndex.drawer + 1,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '-4px 0 10px rgba(0, 0, 0, 0.1)',
  transition: 'transform 0.3s ease-in-out',
  transform: 'translateX(100%)', // Inicialmente oculto
  '&.open': {
    transform: 'translateX(0)', // Visible cuando está abierto
  },
  overflow: 'hidden'
}));

// Componente principal de previsualización
const DocxPreviewPanel = ({ open, onClose, reportPath, reportTitle }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [base64Content, setBase64Content] = useState(null);
  const panelRef = useRef(null);

  // Simular la carga del documento
  useEffect(() => {
    if (open && reportPath) {
      setLoading(true);
      setError(null);

      // En un entorno real, haríamos una solicitud al servidor para obtener el contenido base64
      // Aquí simulamos la carga con un temporizador
      const timer = setTimeout(() => {
        // Simulación de carga exitosa
        if (reportPath.includes('_Normativo_')) {
          // Simulamos que tenemos el contenido base64
          setLoading(false);
          
          // En un entorno real, aquí se cargaría el contenido base64 del servidor
          // En este ejemplo, solo simulamos que tenemos acceso al documento
          convertDocxToHtml(null); // En implementación real, pasaríamos el base64
        } else {
          setError('No se pudo cargar el documento');
          setLoading(false);
        }
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [open, reportPath]);

  // Función para convertir DOCX a HTML usando mammoth
  const convertDocxToHtml = async (base64Data) => {
    try {
      // En una implementación real, convertiríamos el base64 a un ArrayBuffer
      // y lo pasaríamos a mammoth
      
      // Para esta demostración, usamos HTML de ejemplo
      const exampleHtml = `
        <div class="document">
          <h1>Normativa PCI-DSS aplicable a comercios que aceptan tarjetas VISA</h1>
          <div class="date">Fecha: ${new Date().toLocaleDateString()}</div>
          
          <h2>Resumen Ejecutivo</h2>
          <p>Este informe detalla los requisitos normativos de PCI-DSS que deben cumplir los comercios que aceptan tarjetas VISA. La normativa PCI-DSS (Payment Card Industry Data Security Standard) establece un conjunto de requisitos diseñados para garantizar que todas las empresas que procesan, almacenan o transmiten información de tarjetas de crédito mantengan un entorno seguro.</p>
          
          <h2>Marco Regulatorio</h2>
          <p>El estándar PCI-DSS fue desarrollado por el Consejo de Estándares de Seguridad de la Industria de Tarjetas de Pago, fundado por American Express, Discover Financial Services, JCB International, MasterCard y Visa Inc.</p>
          
          <h3>Requisitos Principales</h3>
          <ol>
            <li>Desarrollar y mantener redes y sistemas seguros</li>
            <li>Proteger los datos del titular de la tarjeta</li>
            <li>Mantener un programa de gestión de vulnerabilidades</li>
            <li>Implementar medidas sólidas de control de acceso</li>
            <li>Supervisar y probar regularmente las redes</li>
            <li>Mantener una política de seguridad de la información</li>
          </ol>
          
          <h2>Análisis de Riesgo e Impactos</h2>
          <p>Los comercios que no cumplen con PCI-DSS enfrentan múltiples riesgos, incluyendo multas significativas, aumento en las tarifas de procesamiento, daño reputacional y posible pérdida del derecho a procesar pagos con tarjetas.</p>
          
          <h3>Recomendaciones</h3>
          <ul>
            <li>Realizar una evaluación inicial del entorno de datos de tarjetas</li>
            <li>Implementar controles de seguridad según los requisitos aplicables</li>
            <li>Documentar todos los procesos y políticas de seguridad</li>
            <li>Capacitar al personal en prácticas seguras de manejo de datos</li>
            <li>Realizar auditorías periódicas para mantener la conformidad</li>
          </ul>
        </div>
      `;
      
      setContent(exampleHtml);
      setLoading(false);
    } catch (err) {
      console.error('Error al convertir el documento:', err);
      setError(`Error al procesar el documento: ${err.message}`);
      setLoading(false);
    }
  };

  // Simular descarga del documento
  const handleDownload = () => {
    // En una implementación real, aquí redirigiríamos al endpoint de descarga
    window.open(`/api/report/download/${encodeURIComponent(reportPath)}`, '_blank');
  };

  // Aplicar CSS personalizado para los documentos convertidos
  const documentStyles = `
    .document {
      font-family: 'Arial', sans-serif;
      line-height: 1.6;
      color: #333;
      padding: 1em;
    }
    .document h1 {
      font-size: 1.8em;
      color: #4D0A2E;
      margin-bottom: 0.5em;
    }
    .document h2 {
      font-size: 1.4em;
      color: #4D0A2E;
      margin-top: 1.5em;
      margin-bottom: 0.5em;
      border-bottom: 1px solid #e0e0e0;
      padding-bottom: 0.3em;
    }
    .document h3 {
      font-size: 1.2em;
      color: #333;
      margin-top: 1.2em;
      margin-bottom: 0.5em;
    }
    .document p {
      margin-bottom: 1em;
    }
    .document ul, .document ol {
      margin-bottom: 1em;
      padding-left: 2em;
    }
    .document .date {
      color: #666;
      font-style: italic;
      margin-bottom: 2em;
    }
    .document table {
      border-collapse: collapse;
      width: 100%;
      margin: 1em 0;
    }
    .document table, .document th, .document td {
      border: 1px solid #ddd;
    }
    .document th, .document td {
      padding: 8px;
      text-align: left;
    }
    .document th {
      background-color: #f2f2f2;
    }
  `;

  return (
    <PreviewPanel 
      ref={panelRef}
      className={open ? 'open' : ''}
      elevation={4}
    >
      {/* Cabecera del panel */}
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        p: 2, 
        borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
        backgroundColor: '#4D0A2E',
        color: 'white'
      }}>
        <VisibilityIcon sx={{ mr: 1 }} />
        <Typography variant="h6" sx={{ flexGrow: 1, fontSize: '1.1rem' }}>
          Vista previa del informe
        </Typography>
        <Tooltip title="Descargar documento">
          <IconButton 
            onClick={handleDownload} 
            color="inherit"
            disabled={loading || error}
          >
            <DownloadIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Cerrar vista previa">
          <IconButton onClick={onClose} color="inherit">
            <CloseIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      {/* Información del documento */}
      <Box sx={{ 
        p: 2, 
        backgroundColor: '#f5f5f5', 
        borderBottom: '1px solid rgba(0, 0, 0, 0.12)'
      }}>
        <Typography variant="subtitle1" fontWeight="medium">
          {reportTitle || 'Informe Normativo'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {reportPath?.split('/').pop() || 'Documento sin ruta especificada'}
        </Typography>
      </Box>
      
      {/* Contenido del documento */}
      <Box sx={{ 
        flexGrow: 1, 
        overflow: 'auto',
        bgcolor: 'background.paper',
        position: 'relative',
        p: 0
      }}>
        {/* Estilos CSS para el documento */}
        <style>{documentStyles}</style>
        
        {loading ? (
          <Box 
            display="flex" 
            flexDirection="column"
            alignItems="center" 
            justifyContent="center" 
            height="100%"
            p={4}
          >
            <CircularProgress size={40} sx={{ mb: 2, color: '#4D0A2E' }} />
            <Typography variant="body1" color="text.secondary">
              Cargando documento...
            </Typography>
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ m: 2 }}>
            {error}
          </Alert>
        ) : (
          <Box 
            sx={{ 
              height: '100%',
              overflow: 'auto',
              backgroundColor: '#fff'
            }}
          >
            <div dangerouslySetInnerHTML={{ __html: content }} />
          </Box>
        )}
      </Box>
      
      {/* Pie del panel */}
      <Box sx={{ 
        p: 2, 
        borderTop: '1px solid rgba(0, 0, 0, 0.12)',
        display: 'flex',
        justifyContent: 'space-between',
        bgcolor: 'background.paper'
      }}>
        <Button
          size="small"
          onClick={onClose}
          sx={{ color: 'text.secondary' }}
        >
          Cerrar
        </Button>
        
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          size="small"
          onClick={handleDownload}
          disabled={loading || error}
          sx={{ 
            backgroundColor: '#4D0A2E',
            '&:hover': {
              backgroundColor: '#300621',
            }
          }}
        >
          Descargar
        </Button>
      </Box>
    </PreviewPanel>
  );
};

export default DocxPreviewPanel;