import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Container, 
  Paper, 
  Button, 
  Divider,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  CircularProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ShareIcon from '@mui/icons-material/Share';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { useQueryStore } from '../contexts/store';

const ConsultaHistorialPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { recentQueries, loadStoredQueries } = useQueryStore();
  const [consulta, setConsulta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Función para extraer el texto de la respuesta de manera segura
  const extractResponseText = (response) => {
    // Si la respuesta es null o undefined, devolver mensaje por defecto
    if (!response) {
      return 'No hay respuesta disponible para esta consulta.';
    }

    // Si ya es una cadena de texto, devolverla directamente
    if (typeof response === 'string') {
      return response;
    }

    // Si es un objeto, intentar extraer el texto de diferentes formas
    if (typeof response === 'object') {
      // Buscar la propiedad 'response' que suele contener el texto principal
      if (response.response && typeof response.response === 'string') {
        return response.response;
      }
      
      // Buscar otras propiedades comunes que puedan contener el texto
      if (response.text && typeof response.text === 'string') {
        return response.text;
      }
      
      if (response.content && typeof response.content === 'string') {
        return response.content;
      }
      
      if (response.data && typeof response.data === 'string') {
        return response.data;
      }

      // Si ninguna propiedad conocida contiene texto, intentar convertir a JSON legible
      try {
        return `**Respuesta estructurada:**\n\n\`\`\`json\n${JSON.stringify(response, null, 2)}\n\`\`\``;
      } catch (jsonError) {
        return 'La respuesta contiene datos estructurados que no se pueden mostrar correctamente.';
      }
    }

    // Para cualquier otro tipo de dato, convertir a string de manera segura
    try {
      return String(response);
    } catch (conversionError) {
      return 'Error al procesar la respuesta de esta consulta.';
    }
  };

  // Función para buscar la consulta específica
  useEffect(() => {
    const searchForQuery = async () => {
      try {
        setLoading(true);
        setError(null);

        // Primero asegurar que tenemos las consultas cargadas
        if (recentQueries.length === 0) {
          await loadStoredQueries();
        }

        // Buscar la consulta después de un pequeño delay para asegurar que se cargó
        setTimeout(() => {
          const currentQueries = useQueryStore.getState().recentQueries;
          
          if (id && currentQueries.length > 0) {
            const consultaEncontrada = currentQueries.find(q => q.id === id);
            
            if (consultaEncontrada) {
              setConsulta(consultaEncontrada);
            } else {
              setError(`No se encontró la consulta con ID: ${id}`);
            }
          } else if (currentQueries.length === 0) {
            // Intento directo desde localStorage
            try {
              const stored = localStorage.getItem('recentQueries');
              if (stored) {
                const queries = JSON.parse(stored);
                const consultaEncontrada = queries.find(q => q.id === id);
                if (consultaEncontrada) {
                  setConsulta(consultaEncontrada);
                } else {
                  setError(`Consulta no encontrada en el historial con ID: ${id}`);
                }
              } else {
                setError('No hay consultas guardadas en el historial');
              }
            } catch (localStorageError) {
              console.error('Error al cargar desde localStorage:', localStorageError);
              setError('Error al acceder al historial de consultas');
            }
          } else {
            setError('ID de consulta no válido');
          }
          
          setLoading(false);
        }, 100);

      } catch (err) {
        console.error('Error en searchForQuery:', err);
        setError(`Error al buscar la consulta: ${err.message}`);
        setLoading(false);
      }
    };

    searchForQuery();
  }, [id, recentQueries, loadStoredQueries]);

  // Función para volver atrás
  const handleGoBack = () => {
    navigate('/');
  };

  // Función para compartir la consulta
  const handleShare = async () => {
    if (!consulta) return;

    const responseText = extractResponseText(consulta.response);
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Consulta Normativa',
          text: consulta.text,
          url: window.location.href,
        });
      } catch (error) {
        handleCopyToClipboard(responseText);
      }
    } else {
      handleCopyToClipboard(responseText);
    }
  };

  // Función para copiar al portapapeles
  const handleCopyToClipboard = (responseText) => {
    if (consulta) {
      const textToShare = `Consulta: ${consulta.text}\n\nRespuesta: ${responseText}`;
      navigator.clipboard.writeText(textToShare).then(() => {
        alert('Contenido copiado al portapapeles');
      }).catch(() => {
        alert('No se pudo copiar al portapapeles');
      });
    }
  };

  // Función para formatear la fecha
  const formatDate = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleDateString('es-ES', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (error) {
      return 'Fecha no disponible';
    }
  };

  // Estados de carga
  if (loading) {
    return (
      // Contenedor principal con scroll habilitado
      <Box sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'auto' // Permitir scroll en este contenedor
      }}>
        <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '200px' }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Cargando consulta...</Typography>
          </Box>
        </Container>
      </Box>
    );
  }

  // Estados de error
  if (error) {
    return (
      <Box sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'auto' 
      }}>
        <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
          
          <Button 
            variant="contained" 
            onClick={handleGoBack}
            startIcon={<ArrowBackIcon />}
          >
            Volver a Consultas
          </Button>
        </Container>
      </Box>
    );
  }

  // Estado cuando no se encuentra la consulta
  if (!consulta) {
    return (
      <Box sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'auto' 
      }}>
        <Container maxWidth="lg" sx={{ py: 4, flex: 1 }}>
          <Alert severity="warning" sx={{ mb: 3 }}>
            No se pudo encontrar la consulta solicitada. Puede que haya sido eliminada o el enlace sea incorrecto.
          </Alert>
          
          <Button 
            variant="contained" 
            onClick={handleGoBack}
            sx={{ mt: 2 }}
            startIcon={<ArrowBackIcon />}
          >
            Volver a Consultas
          </Button>
        </Container>
      </Box>
    );
  }

  // Extraer el texto de la respuesta de manera segura
  const responseText = extractResponseText(consulta.response);

  // Render principal con scroll habilitado
  return (
    // El contenedor principal ahora permite scroll y tiene altura mínima adecuada
    <Box sx={{ 
      minHeight: 'calc(100vh - 64px)', // Altura mínima menos la altura del AppBar
      display: 'flex', 
      flexDirection: 'column',
      overflow: 'visible', // Permitir que el contenido se extienda naturalmente
      pb: 4 // Padding bottom para asegurar espacio al final
    }}>
      <Container maxWidth="lg" sx={{ 
        py: 4, 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        gap: 3 // Espaciado consistente entre elementos
      }}>
        {/* Cabecera con navegación - posición fija mejorada */}
        <Paper elevation={1} sx={{ 
          p: 2,
          position: 'sticky', // Mantener la cabecera visible durante el scroll
          top: 0,
          zIndex: 10,
          backgroundColor: 'background.paper',
          borderRadius: 2
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Tooltip title="Volver a consultas">
              <IconButton 
                onClick={handleGoBack}
                sx={{ mr: 2 }}
              >
                <ArrowBackIcon />
              </IconButton>
            </Tooltip>
            
            <Typography variant="h4" sx={{ flexGrow: 1 }}>
              Consulta del Historial
            </Typography>
            
            <Tooltip title="Compartir consulta">
              <span>
                <IconButton 
                  onClick={handleShare}
                  disabled={!consulta}
                >
                  <ShareIcon />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Paper>

        {/* Información de la consulta */}
        <Paper elevation={1} sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Typography variant="h6" color="primary" gutterBottom>
              Consulta Original
            </Typography>
            <Chip 
              label={formatDate(consulta.timestamp)}
              variant="outlined"
              size="small"
            />
          </Box>
          
          <Typography 
            variant="body1" 
            sx={{ 
              p: 2, 
              bgcolor: 'background.default', 
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
              lineHeight: 1.6 // Mejor legibilidad
            }}
          >
            {consulta.text}
          </Typography>
        </Paper>

        <Divider />

        {/* Respuesta - Este es el contenido que puede ser muy largo */}
        <Paper elevation={1} sx={{ 
          p: 3,
          flex: 1, // Permitir que este contenedor crezca según el contenido
          overflow: 'visible' // Asegurar que el contenido largo sea visible
        }}>
          <Typography variant="h6" color="primary" gutterBottom>
            Respuesta del Agente
          </Typography>
          
          <Box 
            className="markdown-content" 
            sx={{ 
              // Estilos de contenido Markdown optimizados para scroll
              '& p': { mt: 0, mb: 2, fontSize: '1.05rem', lineHeight: 1.7 },
              '& ul, & ol': { mt: 0, mb: 2, pl: 3 },
              '& li': { mb: 1 }, // Más espacio entre elementos de lista
              '& code': { 
                p: 0.5, 
                borderRadius: 1, 
                bgcolor: 'rgba(0, 0, 0, 0.04)', 
                fontFamily: 'monospace' 
              },
              '& pre': { 
                p: 1.5, 
                borderRadius: 1, 
                bgcolor: 'rgba(0, 0, 0, 0.04)', 
                overflowX: 'auto',
                mb: 2, // Espaciado después de bloques de código
                '& code': { p: 0, bgcolor: 'transparent' }
              },
              '& blockquote': { 
                borderLeft: '4px solid #e0e0e0', 
                pl: 2, 
                ml: 0, 
                my: 2, // Espaciado vertical para citas
                fontStyle: 'italic' 
              },
              '& h1, & h2, & h3, & h4, & h5, & h6': { 
                mt: 3, // Más espacio antes de títulos
                mb: 1.5 // Más espacio después de títulos
              },
              '& mark': {
                backgroundColor: '#ffeb3b',
                padding: '2px 4px',
                fontSize: '1.4em',
                fontWeight: 500,
                borderRadius: '3px'
              },
              // Estilos para tablas con mejor espaciado
              '& table': {
                width: '100%',
                borderCollapse: 'collapse',
                margin: '24px 0', // Más espacio alrededor de tablas
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                overflow: 'hidden',
                borderRadius: '8px'
              },
              '& thead': {
                backgroundColor: '#4F062A',
                color: 'white'
              },
              '& th': {
                padding: '16px', // Más padding en celdas de encabezado
                textAlign: 'left',
                fontWeight: 600,
                fontSize: '0.9rem',
                borderBottom: 'none'
              },
              '& td': {
                padding: '16px', // Más padding en celdas de datos
                borderBottom: '1px solid #e0e0e0',
                fontSize: '0.9rem',
                verticalAlign: 'top',
                lineHeight: 1.5
              },
              '& tbody tr': {
                backgroundColor: '#fff',
                transition: 'background-color 0.2s ease',
                '&:hover': {
                  backgroundColor: '#f9f9f9'
                },
                '&:last-child td': {
                  borderBottom: 'none'
                }
              },
              '& tbody tr:nth-of-type(even)': {
                backgroundColor: '#f8f9fa'
              },
            }}
          >
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
            >
              {responseText}
            </ReactMarkdown>
          </Box>
        </Paper>

        {/* Acciones adicionales - siempre visibles al final */}
        <Paper elevation={1} sx={{ 
          p: 3,
          mt: 2,
          backgroundColor: 'background.default'
        }}>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
            <Button 
              variant="outlined" 
              onClick={handleGoBack}
              startIcon={<ArrowBackIcon />}
            >
              Volver a Consultas
            </Button>
            
            <Button 
              variant="contained" 
              onClick={handleShare}
              startIcon={<ShareIcon />}
            >
              Compartir Consulta
            </Button>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
};

export default ConsultaHistorialPage;