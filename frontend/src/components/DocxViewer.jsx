// frontend/src/components/DocxViewer.jsx
import React, { useState, useEffect } from 'react';
import { Box, CircularProgress, Typography, Alert } from '@mui/material';
import mammoth from 'mammoth';

/**
 * Componente para mostrar documentos DOCX desde una URL
 * 
 * @param {Object} props - Props del componente
 * @param {string} props.url - URL del documento DOCX
 * @param {string} props.viewerType - Tipo de visualizador ('mammoth' por defecto)
 */
export const DocxViewerFromUrl = ({ url, viewerType = 'mammoth' }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        setLoading(true);

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Error al obtener el documento: ${response.statusText}`);
        }

        // Obtener el blob del documento
        const blob = await response.blob();
        
        if (viewerType === 'mammoth') {
          // Usar mammoth para convertir DOCX a HTML
          const arrayBuffer = await blob.arrayBuffer();
          const result = await mammoth.convertToHtml({ arrayBuffer });
          setContent(result.value);
        } else {
          // Si no es mammoth, probablemente sea una respuesta HTML directa
          const text = await blob.text();
          setContent(text);
        }

        setLoading(false);
      } catch (err) {
        console.error('Error al cargar el documento:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    if (url) {
      fetchDocument();
    }
  }, [url, viewerType]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" p={4}>
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Cargando documento...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error al cargar el documento: {error}
      </Alert>
    );
  }

  return (
    <Box 
      className="document-preview" 
      sx={{ 
        border: '1px solid #ddd',
        borderRadius: 1,
        p: 3,
        bgcolor: '#fff',
        maxHeight: '600px',
        overflowY: 'auto'
      }}
    >
      <div dangerouslySetInnerHTML={{ __html: content }} />
    </Box>
  );
};