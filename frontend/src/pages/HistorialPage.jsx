import React, { useState, useEffect } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Container, 
  Paper, 
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  TextField,
  InputAdornment,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import DeleteIcon from '@mui/icons-material/Delete';
import ShareIcon from '@mui/icons-material/Share';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ClearAllIcon from '@mui/icons-material/ClearAll';
import { useQueryStore } from '../contexts/store';

const HistorialPage = () => {
  const navigate = useNavigate();
  const { recentQueries, loadStoredQueries } = useQueryStore();
  
  // Estados locales para la interfaz
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredQueries, setFilteredQueries] = useState([]);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [queryToDelete, setQueryToDelete] = useState(null);
  const [confirmClearAllOpen, setConfirmClearAllOpen] = useState(false);

  // Cargar consultas al montar el componente
  useEffect(() => {
    loadStoredQueries();
  }, [loadStoredQueries]);

  // Filtrar consultas basado en el término de búsqueda
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredQueries(recentQueries);
    } else {
      const filtered = recentQueries.filter(query =>
        query.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (query.response && query.response.toLowerCase().includes(searchTerm.toLowerCase()))
      );
      setFilteredQueries(filtered);
    }
  }, [recentQueries, searchTerm]);

  // Función para formatear la fecha
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 1) {
      return 'Hace menos de una hora';
    } else if (diffInHours < 24) {
      return `Hace ${Math.floor(diffInHours)} horas`;
    } else if (diffInHours < 48) {
      return 'Ayer';
    } else {
      return date.toLocaleDateString('es-ES', {
        day: 'numeric',
        month: 'short',
        year: 'numeric'
      });
    }
  };

  // Función para truncar texto
  const truncateText = (text, maxLength = 150) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  // Función para eliminar una consulta específica
  const handleDeleteQuery = (queryId) => {
    const queryToRemove = recentQueries.find(q => q.id === queryId);
    setQueryToDelete(queryToRemove);
    setConfirmDeleteOpen(true);
  };

  // Confirmar eliminación de consulta específica
  const confirmDeleteQuery = () => {
    if (queryToDelete) {
      const updatedQueries = recentQueries.filter(q => q.id !== queryToDelete.id);
      
      // Actualizar localStorage
      localStorage.setItem('recentQueries', JSON.stringify(updatedQueries));
      
      // Recargar desde localStorage para actualizar el estado
      loadStoredQueries();
    }
    
    setConfirmDeleteOpen(false);
    setQueryToDelete(null);
  };

  // Función para limpiar todo el historial
  const handleClearAll = () => {
    setConfirmClearAllOpen(true);
  };

  // Confirmar limpieza de todo el historial
  const confirmClearAll = () => {
    localStorage.removeItem('recentQueries');
    loadStoredQueries();
    setConfirmClearAllOpen(false);
  };

  // Función para compartir una consulta
  const handleShareQuery = async (query) => {
    const shareUrl = `${window.location.origin}/consulta/${query.id}`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Consulta Normativa',
          text: truncateText(query.text, 100),
          url: shareUrl,
        });
      } catch (error) {
        // Si falla, copiar al portapapeles
        navigator.clipboard.writeText(shareUrl);
        alert('Enlace copiado al portapapeles');
      }
    } else {
      // Copiar al portapapeles como fallback
      navigator.clipboard.writeText(shareUrl);
      alert('Enlace copiado al portapapeles');
    }
  };

  // Detectar tipo de consulta para mostrar chips diferentes
  const getQueryType = (queryText) => {
    const lowerText = queryText.toLowerCase();
    if (lowerText.includes('gap') || lowerText.includes('análisis')) {
      return { label: 'Análisis GAP', color: 'secondary' };
    } else if (lowerText.includes('visa') || lowerText.includes('mastercard')) {
      return { label: 'Normativa', color: 'primary' };
    } else {
      return { label: 'Consulta', color: 'default' };
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Cabecera */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Historial de Consultas
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Revisa y gestiona tus consultas anteriores
        </Typography>
      </Box>

      {/* Barra de búsqueda y acciones */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField
            placeholder="Buscar en el historial..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
            }}
            sx={{ flexGrow: 1, minWidth: '300px' }}
            variant="outlined"
            size="small"
          />
          
          <Button
            variant="outlined"
            color="error"
            startIcon={<ClearAllIcon />}
            onClick={handleClearAll}
            disabled={recentQueries.length === 0}
            size="small"
          >
            Limpiar Todo
          </Button>
        </Box>
        
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          {filteredQueries.length} de {recentQueries.length} consultas
        </Typography>
      </Paper>

      {/* Lista de consultas */}
      {filteredQueries.length === 0 ? (
        <Paper elevation={1} sx={{ p: 4, textAlign: 'center' }}>
          {recentQueries.length === 0 ? (
            <Alert severity="info">
              No tienes consultas en tu historial aún. ¡Comienza haciendo tu primera consulta!
            </Alert>
          ) : (
            <Alert severity="info">
              No se encontraron consultas que coincidan con tu búsqueda.
            </Alert>
          )}
          
          <Button 
            variant="contained"
            onClick={() => navigate('/')}
            sx={{ mt: 2 }}
          >
            Hacer Nueva Consulta
          </Button>
        </Paper>
      ) : (
        <Paper elevation={1}>
          <List>
            {filteredQueries.map((query, index) => {
              const queryType = getQueryType(query.text);
              
              return (
                <React.Fragment key={query.id}>
                  <ListItem
                    sx={{
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            {truncateText(query.text, 80)}
                          </Typography>
                          <Chip 
                            label={queryType.label} 
                            color={queryType.color}
                            size="small" 
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                            {query.response ? truncateText(query.response, 200) : 'Sin respuesta disponible'}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatDate(query.timestamp)}
                          </Typography>
                        </Box>
                      }
                      onClick={() => navigate(`/consulta/${query.id}`)}
                    />
                    
                    <ListItemSecondaryAction>
                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                        <IconButton
                          size="small"
                          onClick={() => navigate(`/consulta/${query.id}`)}
                          title="Ver consulta completa"
                        >
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                        
                        <IconButton
                          size="small"
                          onClick={() => handleShareQuery(query)}
                          title="Compartir consulta"
                        >
                          <ShareIcon fontSize="small" />
                        </IconButton>
                        
                        <IconButton
                          size="small"
                          onClick={() => handleDeleteQuery(query.id)}
                          title="Eliminar consulta"
                          color="error"
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </ListItemSecondaryAction>
                  </ListItem>
                  
                  {index < filteredQueries.length - 1 && <Divider />}
                </React.Fragment>
              );
            })}
          </List>
        </Paper>
      )}

      {/* Dialog de confirmación para eliminar consulta específica */}
      <Dialog open={confirmDeleteOpen} onClose={() => setConfirmDeleteOpen(false)}>
        <DialogTitle>Confirmar Eliminación</DialogTitle>
        <DialogContent>
          <Typography>
            ¿Estás seguro de que quieres eliminar esta consulta? Esta acción no se puede deshacer.
          </Typography>
          {queryToDelete && (
            <Paper sx={{ p: 2, mt: 2, bgcolor: 'background.default' }}>
              <Typography variant="body2" fontWeight="medium">
                {truncateText(queryToDelete.text, 100)}
              </Typography>
            </Paper>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDeleteOpen(false)}>
            Cancelar
          </Button>
          <Button onClick={confirmDeleteQuery} color="error" variant="contained">
            Eliminar
          </Button>
        </DialogActions>
      </Dialog>

      {/* Dialog de confirmación para limpiar todo */}
      <Dialog open={confirmClearAllOpen} onClose={() => setConfirmClearAllOpen(false)}>
        <DialogTitle>Limpiar Todo el Historial</DialogTitle>
        <DialogContent>
          <Typography>
            ¿Estás seguro de que quieres eliminar todo tu historial de consultas? 
            Esta acción eliminará permanentemente todas las consultas guardadas y no se puede deshacer.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmClearAllOpen(false)}>
            Cancelar
          </Button>
          <Button onClick={confirmClearAll} color="error" variant="contained">
            Limpiar Todo
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default HistorialPage;