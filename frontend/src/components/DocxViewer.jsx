import React, { useEffect, useRef, useState } from 'react';
import { Box, CircularProgress, Paper, Typography } from '@mui/material';
import mammoth from 'mammoth';
import { renderAsync } from 'docx-preview';

const DocxViewer = ({ docxFile, viewerType = 'mammoth' }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!docxFile) {
      setLoading(false);
      return;
    }

    const loadDocument = async () => {
      try {
        setLoading(true);
        setError(null);

        if (viewerType === 'mammoth') {
          // Usar mammoth.js para convertir DOCX a HTML
          const arrayBuffer = await docxFile.arrayBuffer();
          const result = await mammoth.convertToHtml({ arrayBuffer });
          
          if (containerRef.current) {
            containerRef.current.innerHTML = result.value;
          }
        } else if (viewerType === 'docx-preview') {
          // Usar docx-preview para renderizar DOCX
          if (containerRef.current) {
            // Limpiar el contenedor primero
            containerRef.current.innerHTML = '';
            
            const arrayBuffer = await docxFile.arrayBuffer();
            await renderAsync(arrayBuffer, containerRef.current, null, {
              inWrapper: true,
              ignoreWidth: false,
              ignoreHeight: false,
              ignoreFonts: false,
              breakPages: true,
              debugLog: false,
              experimental: false,
            });
          }
        }
      } catch (err) {
        console.error('Error al visualizar el documento Word:', err);
        setError('No se pudo cargar el documento. Por favor, inténtelo de nuevo.');
      } finally {
        setLoading(false);
      }
    };

    loadDocument();
  }, [docxFile, viewerType]);

  return (
    <Paper elevation={3} sx={{ p: 2, my: 2, minHeight: '400px' }}>
      {loading && (
        <Box display="flex" justifyContent="center" alignItems="center" height="400px">
          <CircularProgress />
        </Box>
      )}
      
      {error && (
        <Box display="flex" justifyContent="center" alignItems="center" height="400px">
          <Typography color="error">{error}</Typography>
        </Box>
      )}
      
      <Box 
        ref={containerRef} 
        sx={{ 
          '& img': {
            maxWidth: '100%'
          },
          '& table': {
            borderCollapse: 'collapse',
            width: '100%',
            mb: 2
          },
          '& td, & th': {
            border: '1px solid #ddd',
            padding: '8px'
          },
          '& tr:nth-of-type(odd)': {
            backgroundColor: '#f9f9f9'
          },
          '& th': {
            paddingTop: '12px',
            paddingBottom: '12px',
            textAlign: 'left',
            backgroundColor: '#f2f2f2',
            color: 'black'
          }
        }}
      />
    </Paper>
  );
};

// Componente para cargar y visualizar un documento DOCX desde una URL
export const DocxViewerFromUrl = ({ url, viewerType = 'mammoth' }) => {
  const [docxFile, setDocxFile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) {
      setLoading(false);
      return;
    }

    const fetchDocx = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Error al obtener el documento: ${response.status}`);
        }
        
        const blob = await response.blob();
        const file = new File([blob], 'document.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
        
        setDocxFile(file);
      } catch (err) {
        console.error('Error al obtener el documento:', err);
        setError('No se pudo cargar el documento. Por favor, inténtelo de nuevo.');
      } finally {
        setLoading(false);
      }
    };

    fetchDocx();
  }, [url]);

  if (loading && !docxFile) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return docxFile ? <DocxViewer docxFile={docxFile} viewerType={viewerType} /> : null;
};

export default DocxViewer;