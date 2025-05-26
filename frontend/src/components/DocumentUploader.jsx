import React, { useState, useCallback } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Alert, 
  LinearProgress,
  IconButton,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import DescriptionIcon from '@mui/icons-material/Description';
import { styled } from '@mui/material/styles';

// Componente estilizado para el área de drag and drop
const UploadArea = styled(Paper)(({ theme, isDragActive }) => ({
  padding: theme.spacing(3),
  textAlign: 'center',
  cursor: 'pointer',
  border: `2px dashed ${isDragActive ? theme.palette.primary.main : theme.palette.grey[300]}`,
  backgroundColor: isDragActive ? theme.palette.primary.light + '10' : theme.palette.background.default,
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: theme.palette.primary.light + '05',
  }
}));

const DocumentUploader = ({ onFilesChange, maxFiles = 5, allowedTypes = ['.pdf', '.docx', '.txt'] }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  // Función para validar archivos
  const validateFile = (file) => {
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (file.size > maxSize) {
      return 'El archivo es demasiado grande. Máximo 10MB.';
    }
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(extension)) {
      return `Tipo de archivo no permitido. Tipos válidos: ${allowedTypes.join(', ')}`;
    }
    
    return null;
  };

  // Función para procesar archivos seleccionados
  const processFiles = useCallback(async (fileList) => {
    const files = Array.from(fileList);
    
    if (uploadedFiles.length + files.length > maxFiles) {
      setError(`Máximo ${maxFiles} archivos permitidos`);
      return;
    }

    setError(null);
    setIsUploading(true);
    setUploadProgress(0);

    const validFiles = [];
    
    for (const file of files) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        setIsUploading(false);
        return;
      }
      
      // Simular progreso de carga
      setUploadProgress((prev) => prev + (100 / files.length));
      
      // Crear objeto de archivo con metadatos
      const fileObject = {
        id: Date.now() + Math.random(),
        file: file,
        name: file.name,
        size: file.size,
        type: file.type,
        uploadedAt: new Date()
      };
      
      validFiles.push(fileObject);
    }

    // Simular tiempo de procesamiento
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const newFiles = [...uploadedFiles, ...validFiles];
    setUploadedFiles(newFiles);
    setIsUploading(false);
    setUploadProgress(0);
    
    // Notificar al componente padre sobre los cambios
    if (onFilesChange) {
      onFilesChange(newFiles);
    }
  }, [uploadedFiles, maxFiles, allowedTypes, onFilesChange]);

  // Manejadores de drag and drop
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
  }, [processFiles]);

  // Manejador para selección de archivos por click
  const handleFileSelect = useCallback((e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
    // Limpiar el input para permitir cargar el mismo archivo de nuevo
    e.target.value = '';
  }, [processFiles]);

  // Función para eliminar archivo
  const removeFile = useCallback((fileId) => {
    const newFiles = uploadedFiles.filter(f => f.id !== fileId);
    setUploadedFiles(newFiles);
    
    if (onFilesChange) {
      onFilesChange(newFiles);
    }
  }, [uploadedFiles, onFilesChange]);

  // Función para formatear el tamaño del archivo
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box>
      {/* Área de carga */}
      <UploadArea
        isDragActive={isDragActive}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload-input').click()}
      >
        <input
          id="file-upload-input"
          type="file"
          multiple
          accept={allowedTypes.join(',')}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Suelta los archivos aquí' : 'Arrastra archivos aquí o haz click para seleccionar'}
        </Typography>
        
        <Typography variant="body2" color="text.secondary">
          Tipos permitidos: {allowedTypes.join(', ')} | Máximo {maxFiles} archivos | Tamaño máximo: 10MB
        </Typography>
        
        <Button
          variant="contained"
          startIcon={<CloudUploadIcon />}
          sx={{ mt: 2 }}
          onClick={(e) => e.stopPropagation()}
        >
          Seleccionar archivos
        </Button>
      </UploadArea>

      {/* Barra de progreso */}
      {isUploading && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            Procesando archivos...
          </Typography>
          <LinearProgress variant="determinate" value={uploadProgress} />
        </Box>
      )}

      {/* Mensajes de error */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {/* Lista de archivos cargados */}
      {uploadedFiles.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Documentos cargados ({uploadedFiles.length})
          </Typography>
          
          <List>
            {uploadedFiles.map((fileObj) => (
              <ListItem key={fileObj.id} divider>
                <DescriptionIcon sx={{ mr: 2, color: 'primary.main' }} />
                <ListItemText
                  primary={fileObj.name}
                  secondary={`${formatFileSize(fileObj.size)} • Cargado: ${fileObj.uploadedAt.toLocaleTimeString()}`}
                />
                <ListItemSecondaryAction>
                  <Chip 
                    label={fileObj.name.split('.').pop().toUpperCase()} 
                    size="small" 
                    color="primary" 
                    variant="outlined"
                    sx={{ mr: 1 }}
                  />
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => removeFile(fileObj.id)}
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Box>
  );
};

export default DocumentUploader;