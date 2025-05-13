// frontend/src/components/DocxPreviewPanel.jsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  IconButton, 
  CircularProgress,
  Button,
  Divider,
  Tooltip,
  Alert,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  TextField,
  Snackbar,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import HighlightIcon from '@mui/icons-material/Highlight';
import FormatUnderlinedIcon from '@mui/icons-material/FormatUnderlined';
import CommentIcon from '@mui/icons-material/Comment';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import FormatBoldIcon from '@mui/icons-material/FormatBold';
import FormatItalicIcon from '@mui/icons-material/FormatItalic';
import DeleteIcon from '@mui/icons-material/Delete';
import { styled } from '@mui/material/styles';
import { reportService, reportServiceWithFallback } from '../services/api';
import mammoth from 'mammoth';

// Componente estilizado para el panel lateral
const PreviewPanel = styled(Paper)(({ theme }) => ({
  position: 'fixed',
  right: 0,
  top: 64,
  bottom: 0,
  width: '40%',
  maxWidth: 600,
  zIndex: theme.zIndex.drawer + 1,
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '-4px 0 10px rgba(0, 0, 0, 0.1)',
  transition: 'transform 0.3s ease-in-out',
  transform: 'translateX(100%)',
  '&.open': {
    transform: 'translateX(0)',
  },
  overflow: 'hidden'
}));

// Overlay para oscurecer el fondo cuando el panel está abierto
const Overlay = styled(Box)(({ theme }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  zIndex: theme.zIndex.drawer,
  opacity: 0,
  visibility: 'hidden',
  transition: 'opacity 0.3s ease-in-out, visibility 0.3s ease-in-out',
  '&.open': {
    opacity: 1,
    visibility: 'visible',
  }
}));

// Toolbar flotante para formato de texto
const FloatingToolbar = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  zIndex: 1000,
  padding: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  background: theme.palette.background.paper,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5)
}));

// Contenedor para los comentarios
const CommentsContainer = styled(Box)(({ theme }) => ({
  position: 'fixed',
  right: 0,
  top: 64,
  bottom: 0,
  width: '25%',
  maxWidth: 350,
  backgroundColor: theme.palette.background.paper,
  borderLeft: `1px solid ${theme.palette.divider}`,
  zIndex: theme.zIndex.drawer + 2,
  overflowY: 'auto',
  padding: theme.spacing(2),
  transform: 'translateX(100%)',
  transition: 'transform 0.3s ease-in-out',
  '&.open': {
    transform: 'translateX(0)',
  }
}));

// Componente principal de previsualización
const DocxPreviewPanel = ({ open, onClose, reportPath, reportTitle }) => {
  // Estados para manejo del documento
  const [content, setContent] = useState('');
  const [originalContent, setOriginalContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fallbackUsed, setFallbackUsed] = useState(false);

  // Referencias a elementos del DOM
  const panelRef = useRef(null);
  const overlayRef = useRef(null);
  const contentRef = useRef(null);
  const toolbarRef = useRef(null);
  
  // Estados para edición y anotaciones
  const [isEditing, setIsEditing] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [selectionRange, setSelectionRange] = useState(null);
  const [annotationColor, setAnnotationColor] = useState('#FFFF00');
  const [annotations, setAnnotations] = useState([]);
  const [showComments, setShowComments] = useState(false);
  const [reportId, setReportId] = useState('');
  
  // Estados para diálogos y menús
  const [commentDialogOpen, setCommentDialogOpen] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [colorMenuAnchor, setColorMenuAnchor] = useState(null);
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  // Posición de la barra de formato flotante
  const [toolbarPosition, setToolbarPosition] = useState({ top: 0, left: 0 });
  const [showToolbar, setShowToolbar] = useState(false);

  // Colores predefinidos para anotaciones
  const annotationColors = [
    '#FFFF00', // Amarillo
    '#FF9999', // Rojo claro
    '#99FF99', // Verde claro
    '#9999FF', // Azul claro
    '#FF99FF'  // Rosa claro
  ];

  // Extraer ID del reporte de la ruta
  useEffect(() => {
    if (reportPath) {
      const idMatch = reportPath.match(/Reporte_Normativo_(\d+_\d+)\.docx/);
      if (idMatch && idMatch[1]) {
        setReportId(idMatch[1]);
      }
    }
  }, [reportPath]);

  // Cargar el documento cuando se abre el panel
  useEffect(() => {
    if (open && reportPath) {
      setLoading(true);
      setError(null);
      setFallbackUsed(false);
      setIsEditing(false);
      setHasUnsavedChanges(false);
      setShowToolbar(false);
      setShowComments(false);
      setAnnotations([]);

      const fetchDocument = async () => {
        try {
          // Extraer el ID del reporte de la ruta
          const idMatch = reportPath.match(/Reporte_Normativo_(\d+_\d+)\.docx/);
          
          if (idMatch && idMatch[1]) {
            const reportId = idMatch[1];
            console.log("Intentando obtener documento con ID:", reportId);
            
            try {
              // Intentar obtener el contenido real del documento
              const response = await reportService.getReportContentById(reportId);
              console.log("Respuesta del servidor:", response);
              
              if (response && response.base64Content) {
                console.log("Base64 recibido, convirtiendo documento...");
                await convertDocxToHtml(response.base64Content);
                
                // Cargar anotaciones existentes si las hay
                try {
                  const savedAnnotations = await reportService.getAnnotations(reportId);
                  if (savedAnnotations && savedAnnotations.length) {
                    setAnnotations(savedAnnotations);
                  }
                } catch (annotError) {
                  console.warn("No se pudieron cargar las anotaciones:", annotError);
                }
                
                return;
              } else {
                console.warn("Respuesta recibida pero sin contenido base64");
                throw new Error("No se recibieron datos base64 válidos");
              }
            } catch (apiError) {
              console.warn("Error en la API, utilizando modo fallback:", apiError);
              throw apiError;
            }
          } else {
            console.warn("No se pudo extraer ID de la ruta:", reportPath);
            throw new Error("Formato de ruta no reconocido");
          }
        } catch (err) {
          console.error("Error al obtener documento:", err);
          setError(`No se pudo obtener el contenido base64 del documento`);
          setFallbackUsed(true);
          generateDemoContent(reportPath);
        }
      };

      // Simular un pequeño retraso para mejorar UX
      const timer = setTimeout(() => {
        fetchDocument();
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [open, reportPath]);

  // Función para convertir DOCX a HTML usando mammoth
  const convertDocxToHtml = async (base64Data) => {
    try {
      console.log("Iniciando conversión de DOCX a HTML");
      
      // Verificar que tenemos datos válidos
      if (!base64Data || typeof base64Data !== 'string') {
        throw new Error("Datos base64 inválidos o faltantes");
      }
      
      // Convertir base64 a ArrayBuffer
      try {
        const binaryString = window.atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const arrayBuffer = bytes.buffer;
        
        console.log("ArrayBuffer creado, convirtiendo con mammoth...");
        
        // Usar mammoth para convertir DOCX a HTML
        const result = await mammoth.convertToHtml({ 
          arrayBuffer,
          styleMap: [
            "b => strong",
            "i => em",
            "u => u"
          ]
        });
        
        console.log("Conversión completada");
        
        // Procesar el HTML para corregir los formatos
        let processedHtml = result.value;
        
        // Paso 1: Reemplazar patrones de títulos con asteriscos
        processedHtml = processedHtml.replace(/(\d+\.\s+)\*\*(.*?)\*\*/g, '$1<strong>$2</strong>');
        
        // Paso 2: Reemplazar los elementos de lista con asteriscos
        processedHtml = processedHtml.replace(/<li>\*\*(.*?):\*\*\s*(.*?)<\/li>/g, '<li><strong>$1:</strong> $2</li>');
        processedHtml = processedHtml.replace(/<li>\*\*(.*?)\*\*\s*(.*?)<\/li>/g, '<li><strong>$1</strong> $2</li>');
        
        // Paso 3: Eliminar cualquier asterisco restante y aplicar formato
        processedHtml = processedHtml.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Paso 4: Aplicar negrita a los títulos numerados si no se ha aplicado ya
        processedHtml = processedHtml.replace(/(<p>|^)(\d+\.\s+)([^<]+)(<\/p>|$)/g, '$1$2<strong>$3</strong>$4');
        
        // Guardar el contenido original y el procesado
        setOriginalContent(processedHtml);
        setContent(processedHtml);
        setLoading(false);
        
        // Registrar cualquier advertencia de la conversión
        if (result.messages && result.messages.length > 0) {
          console.log("Advertencias de conversión:", result.messages);
        }
      } catch (conversionError) {
        console.error('Error durante la conversión:', conversionError);
        throw new Error(`Error al convertir documento: ${conversionError.message}`);
      }
    } catch (err) {
      console.error('Error al procesar el documento:', err);
      setError(`Error al procesar el documento: ${err.message}`);
      setFallbackUsed(true);
      generateDemoContent(reportPath);
    }
  };

  // Función de fallback para generar contenido de demostración
  const generateDemoContent = (path) => {
    // Extraer información relevante del nombre del archivo/ruta
    let docType = "Normativo";
    let fileDate = new Date().toLocaleDateString();
    
    // Intentar extraer fecha del nombre del archivo
    if (path) {
      const dateMatch = path.match(/(\d{8})_(\d{6})/);
      if (dateMatch) {
        const dateStr = dateMatch[1];
        // Formatear fecha YYYYMMDD a DD/MM/YYYY
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        fileDate = `${day}/${month}/${year}`;
      }
    }

    const demoHtml = `
      <div class="document">
        <h1>Informe Normativo</h1>
        <div class="date">Fecha: ${fileDate}</div>
        
        <h2>1. Introducción</h2>
        <p>Este documento presenta un análisis detallado de la normativa aplicable y los requisitos de cumplimiento para las entidades reguladas.</p>
        
        <h2>2. Marco Regulatorio</h2>
        <p>El marco regulatorio establece las directrices y obligaciones que deben cumplir las entidades para garantizar la seguridad, transparencia y protección de los datos.</p>
        
        <h3>2.1 Objetivos Principales</h3>
        <ul>
          <li><strong>Seguridad:</strong> Garantizar la protección de datos sensibles</li>
          <li><strong>Transparencia:</strong> Asegurar prácticas comerciales claras y justas</li>
          <li><strong>Cumplimiento:</strong> Establecer estándares uniformes en el sector</li>
        </ul>
        
        <h2>3. Requisitos de Cumplimiento</h2>
        <ol>
          <li><strong>Protección de Datos:</strong> Implementar medidas técnicas y organizativas adecuadas</li>
          <li><strong>Autenticación:</strong> Utilizar métodos seguros para verificar la identidad</li>
          <li><strong>Monitoreo:</strong> Realizar seguimiento continuo de las transacciones</li>
          <li><strong>Reportes:</strong> Presentar informes periódicos a las autoridades reguladoras</li>
        </ol>
        
        <h2>4. Recomendaciones</h2>
        <p>Se recomienda implementar los siguientes controles y procedimientos para asegurar el cumplimiento normativo:</p>
        <ul>
          <li>Realizar evaluaciones periódicas de riesgos</li>
          <li>Actualizar regularmente los sistemas de seguridad</li>
          <li>Capacitar al personal en las nuevas normativas</li>
          <li>Documentar todos los procesos y controles implementados</li>
        </ul>
        
        <h2>5. Conclusión</h2>
        <p>El cumplimiento de estas normativas no solo es una obligación legal, sino también una oportunidad para fortalecer la seguridad y confianza en las operaciones comerciales.</p>
      </div>
    `;
    
    setOriginalContent(demoHtml);
    setContent(demoHtml);
    setLoading(false);
  };

  // Función para capturar selección de texto
  const handleTextSelection = () => {
    if (!isEditing) return;
    
    const selection = window.getSelection();
    if (selection.toString().length > 0) {
      setSelectedText(selection.toString());
      
      // Guardar el rango de selección para aplicar formato
      if (selection.rangeCount > 0) {
        setSelectionRange(selection.getRangeAt(0));
        
        // Posicionar la barra de herramientas cerca de la selección
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const contentRect = contentRef.current.getBoundingClientRect();
        
        const position = {
          top: rect.top - contentRect.top - 40, // 40px arriba de la selección
          left: rect.left + (rect.width / 2) - 100 // centrado horizontalmente
        };
        
        setToolbarPosition(position);
        setShowToolbar(true);
      }
    } else {
      setSelectedText('');
      setSelectionRange(null);
      setShowToolbar(false);
    }
  };

  // Función para aplicar formato al texto seleccionado
  const applyTextFormat = (format) => {
    if (!selectionRange) return;
    
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(selectionRange);
    
    // Aplicar el formato según el tipo
    switch (format) {
      case 'bold':
        document.execCommand('bold', false, null);
        break;
      case 'italic':
        document.execCommand('italic', false, null);
        break;
      case 'highlight':
        document.execCommand('hiliteColor', false, annotationColor);
        // Guardar anotación
        addAnnotation('highlight');
        break;
      case 'underline':
        document.execCommand('underline', false, null);
        break;
      default:
        break;
    }
    
    // Actualizar contenido y marcar cambios
    setContent(contentRef.current.innerHTML);
    setHasUnsavedChanges(true);
    
    // Limpiar selección
    setSelectedText('');
    setSelectionRange(null);
    setShowToolbar(false);
  };

  // Función para abrir diálogo de comentario
  const openCommentDialog = () => {
    if (!selectionRange) return;
    setCommentDialogOpen(true);
  };

  // Función para añadir un comentario
  const addComment = () => {
    if (!selectionRange || !commentText.trim()) {
      setCommentDialogOpen(false);
      return;
    }
    
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(selectionRange);
    
    // Crear un elemento span para resaltar el texto comentado
    const span = document.createElement('span');
    span.style.backgroundColor = annotationColor;
    span.style.cursor = 'pointer';
    span.className = 'annotated-text';
    
    // Generar un ID único para el comentario
    const commentId = Date.now().toString();
    span.dataset.commentId = commentId;
    
    // Wrap the selected text with the span
    try {
      // Intentar preservar el range original
      const range = selectionRange.cloneRange();
      range.surroundContents(span);
      
      // Agregar el comentario al estado
      const newAnnotation = {
        id: commentId,
        text: selectedText,
        comment: commentText,
        color: annotationColor,
        timestamp: new Date(),
        type: 'comment'
      };
      
      setAnnotations([...annotations, newAnnotation]);
      setHasUnsavedChanges(true);
      
      // Actualizar el contenido
      setContent(contentRef.current.innerHTML);
    } catch (e) {
      console.error('Error al aplicar comentario:', e);
    }
    
    // Cerrar diálogo y limpiar
    setCommentDialogOpen(false);
    setCommentText('');
    setSelectedText('');
    setSelectionRange(null);
    setShowToolbar(false);
  };

  // Función para añadir una anotación
  const addAnnotation = (type) => {
    if (!selectionRange) return;
    
    // Guardar anotación en el estado
    const newAnnotation = {
      id: Date.now().toString(),
      text: selectedText,
      color: annotationColor,
      timestamp: new Date(),
      type: type
    };
    
    setAnnotations([...annotations, newAnnotation]);
    setHasUnsavedChanges(true);
  };

  // Función para eliminar una anotación
  const removeAnnotation = (id) => {
    // Eliminar del estado
    setAnnotations(annotations.filter(ann => ann.id !== id));
    setHasUnsavedChanges(true);
    
    // Eliminar del DOM si es posible
    const element = document.querySelector(`[data-comment-id="${id}"]`);
    if (element) {
      // Preservar el texto dentro
      const text = element.textContent;
      element.parentNode.replaceChild(document.createTextNode(text), element);
      
      // Actualizar contenido
      setContent(contentRef.current.innerHTML);
    }
  };

  // Función para abrir el menú de colores
  const handleColorMenuOpen = (event) => {
    setColorMenuAnchor(event.currentTarget);
  };

  // Función para cerrar el menú de colores
  const handleColorMenuClose = () => {
    setColorMenuAnchor(null);
  };

  // Función para seleccionar un color
  const selectColor = (color) => {
    setAnnotationColor(color);
    handleColorMenuClose();
  };

  // Función para habilitar/deshabilitar el modo edición
  const toggleEditMode = () => {
    if (isEditing && hasUnsavedChanges) {
      // Preguntar antes de salir del modo edición con cambios sin guardar
      if (window.confirm('Hay cambios sin guardar. ¿Desea guardarlos antes de salir del modo edición?')) {
        saveChanges();
      }
    }
    
    setIsEditing(!isEditing);
    setShowToolbar(false);
    
    // Si estamos saliendo del modo edición sin guardar
    if (isEditing && !hasUnsavedChanges) {
      // Restaurar contenido original si no se guardaron cambios
      setContent(originalContent);
    }
  };

  // Función para guardar cambios
  const saveChanges = async () => {
    try {
      setLoading(true);
      
      // Guardar el contenido modificado
      const modifiedContent = contentRef.current.innerHTML;
      
      // Guardar anotaciones si las hay
      if (annotations.length > 0) {
        await reportService.saveAnnotations(reportId, annotations);
      }
      
      // Actualizar el estado
      setOriginalContent(modifiedContent);
      setHasUnsavedChanges(false);
      
      setNotification({
        open: true,
        message: 'Cambios guardados correctamente',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error al guardar cambios:', error);
      setNotification({
        open: true,
        message: 'Error al guardar los cambios',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Manejar la descarga del documento
  const handleDownload = () => {
    if (reportPath) {
      // Extraer el ID del reporte
      const idMatch = reportPath.match(/Reporte_Normativo_(\d+_\d+)\.docx/);
      if (idMatch && idMatch[1]) {
        const reportId = idMatch[1];
        
        // Usar el ID para la descarga
        window.open(`/api/report/content_by_id/${reportId}/download`, '_blank');
      } else {
        // Intentar usar la ruta directa como fallback
        window.open(`/api/report/download?path=${encodeURIComponent(reportPath)}`, '_blank');
      }
    }
  };

  // Manejar cierre del panel cuando se hace clic en el overlay
  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) {
      // Si hay cambios sin guardar, preguntar antes de cerrar
      if (hasUnsavedChanges) {
        if (window.confirm('Hay cambios sin guardar. ¿Desea guardarlos antes de cerrar?')) {
          saveChanges();
        }
      }
      onClose();
    }
  };

  // Manejar cierre de la notificación
  const handleCloseNotification = () => {
    setNotification({...notification, open: false});
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
    
    .document li {
      margin-bottom: 0.5em;
    }
    
    .document li strong {
      color: #4D0A2E;
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
    
    .document strong, .document b {
      font-weight: bold;
      color: #4D0A2E;
    }
    
    .document em, .document i {
      font-style: italic;
    }
    
    .document u {
      text-decoration: underline;
    }
    
    .annotated-text {
      border-radius: 2px;
      position: relative;
    }
    
    .annotated-text:hover::after {
      content: "💬";
      position: absolute;
      top: -15px;
      right: -10px;
      font-size: 16px;
    }
  `;

  return (
    <>
      {/* Overlay para oscurecer el fondo cuando el panel está abierto */}
      <Overlay 
        ref={overlayRef}
        className={open ? 'open' : ''}
        onClick={handleOverlayClick}
      />
      
      {/* Panel de previsualización */}
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
          
          {/* Botón para alternar comentarios */}
          <Tooltip title={showComments ? "Ocultar comentarios" : "Mostrar comentarios"}>
            <IconButton 
              onClick={() => setShowComments(!showComments)}
              color="inherit"
              disabled={loading || annotations.length === 0}
            >
              <CommentIcon />
              {annotations.length > 0 && (
                <Typography 
                  variant="caption" 
                  sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    right: 0, 
                    backgroundColor: 'error.main',
                    color: 'white',
                    borderRadius: '50%',
                    width: '18px',
                    height: '18px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  {annotations.length}
                </Typography>
              )}
            </IconButton>
          </Tooltip>
          
          {/* Botón para alternar modo edición */}
          <Tooltip title={isEditing ? "Finalizar edición" : "Editar documento"}>
            <IconButton 
              onClick={toggleEditMode}
              color="inherit"
              disabled={loading || error}
            >
              <EditIcon />
            </IconButton>
          </Tooltip>
          
          {/* Botón para guardar cambios */}
          {isEditing && (
            <Tooltip title="Guardar cambios">
              <IconButton 
                onClick={saveChanges}
                color="inherit"
                disabled={!hasUnsavedChanges || loading}
              >
                <SaveIcon />
              </IconButton>
            </Tooltip>
          )}
          
          {/* Botón para descargar */}
          <Tooltip title="Descargar documento">
            <IconButton 
              onClick={handleDownload} 
              color="inherit"
              disabled={loading || error}
            >
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          
          {/*{/* Botón para cerrar */}
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
            {fallbackUsed && (
              <Box component="span" sx={{ color: 'warning.main', ml: 1, fontStyle: 'italic' }}>
                (vista previa generada)
              </Box>
            )}
          </Typography>
          {isEditing && (
            <Typography variant="caption" color="primary" sx={{ display: 'block', mt: 1 }}>
              Modo edición activo. Seleccione texto para aplicar formato.
            </Typography>
          )}
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
                {isEditing && hasUnsavedChanges ? "Guardando cambios..." : "Cargando documento..."}
              </Typography>
            </Box>
          ) : error ? (
            <Box>
              <Alert severity="error" sx={{ m: 2 }}>
                {error}
              </Alert>
              {content && (
                <Box 
                  sx={{ 
                    height: 'calc(100% - 80px)',
                    overflow: 'auto',
                    backgroundColor: '#fff',
                    mt: 2
                  }}
                >
                  <div dangerouslySetInnerHTML={{ __html: content }} />
                </Box>
              )}
            </Box>
          ) : (
            <Box 
              ref={contentRef}
              sx={{ 
                height: '100%',
                overflow: 'auto',
                backgroundColor: '#fff',
                outline: isEditing ? '2px solid rgba(77, 10, 46, 0.2)' : 'none',
                transition: 'outline 0.3s ease'
              }}
              contentEditable={isEditing}
              suppressContentEditableWarning={true}
              onMouseUp={handleTextSelection}
              onInput={() => {
                if (isEditing) {
                  setHasUnsavedChanges(true);
                  setContent(contentRef.current.innerHTML);
                }
              }}
              dangerouslySetInnerHTML={{ __html: content }}
            />
          )}
          
          {/* Barra de herramientas flotante */}
          {showToolbar && (
            <FloatingToolbar
              ref={toolbarRef}
              style={{
                top: `${toolbarPosition.top}px`,
                left: `${toolbarPosition.left}px`
              }}
            >
              <Tooltip title="Negrita">
                <IconButton size="small" onClick={() => applyTextFormat('bold')}>
                  <FormatBoldIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Cursiva">
                <IconButton size="small" onClick={() => applyTextFormat('italic')}>
                  <FormatItalicIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Subrayar">
                <IconButton size="small" onClick={() => applyTextFormat('underline')}>
                  <FormatUnderlinedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Resaltar">
                <IconButton 
                  size="small" 
                  onClick={() => applyTextFormat('highlight')}
                  sx={{ color: annotationColor }}
                >
                  <HighlightIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Comentar">
                <IconButton size="small" onClick={openCommentDialog}>
                  <CommentIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Elegir color">
                <IconButton size="small" onClick={handleColorMenuOpen}>
                  <ColorLensIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </FloatingToolbar>
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
          
          {isEditing ? (
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              size="small"
              onClick={saveChanges}
              disabled={!hasUnsavedChanges || loading}
              sx={{ 
                backgroundColor: '#4D0A2E',
                '&:hover': {
                  backgroundColor: '#300621',
                }
              }}
            >
              Guardar
            </Button>
          ) : (
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
          )}
        </Box>
      </PreviewPanel>
      
      {/* Panel de comentarios */}
      <CommentsContainer className={showComments ? 'open' : ''}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Comentarios
          </Typography>
          <IconButton onClick={() => setShowComments(false)} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
        
        <Divider sx={{ mb: 2 }} />
        
        {annotations.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
            No hay comentarios en este documento.
          </Typography>
        ) : (
          annotations.map((annotation) => (
            <Paper
              key={annotation.id}
              elevation={1}
              sx={{ 
                p: 2, 
                mb: 2, 
                borderLeft: `4px solid ${annotation.color}`,
                backgroundColor: `${annotation.color}20`
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  {annotation.timestamp ? new Date(annotation.timestamp).toLocaleString() : 'Sin fecha'}
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => removeAnnotation(annotation.id)}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>
              
              <Typography variant="body2" fontWeight="medium" sx={{ mb: 1 }}>
                "{annotation.text}"
              </Typography>
              
              {annotation.comment && (
                <Typography variant="body2">
                  {annotation.comment}
                </Typography>
              )}
            </Paper>
          ))
        )}
      </CommentsContainer>
      
      {/* Diálogo para añadir comentarios */}
      <Dialog open={commentDialogOpen} onClose={() => setCommentDialogOpen(false)}>
        <DialogTitle>Añadir comentario</DialogTitle>
        <DialogContent>
          <Typography variant="subtitle2" gutterBottom>
            Texto seleccionado:
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              p: 1, 
              mb: 2, 
              backgroundColor: `${annotationColor}50`,
              borderLeft: `3px solid ${annotationColor}`
            }}
          >
            "{selectedText}"
          </Typography>
          
          <TextField
            autoFocus
            margin="dense"
            id="comment"
            label="Comentario"
            type="text"
            fullWidth
            multiline
            rows={4}
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCommentDialogOpen(false)}>Cancelar</Button>
          <Button onClick={addComment} variant="contained" color="primary">
            Guardar
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Menú de colores */}
      <Menu
        anchorEl={colorMenuAnchor}
        open={Boolean(colorMenuAnchor)}
        onClose={handleColorMenuClose}
      >
        {annotationColors.map((color) => (
          <MenuItem key={color} onClick={() => selectColor(color)}>
            <ListItemIcon>
              <Box 
                sx={{ 
                  width: 20, 
                  height: 20, 
                  borderRadius: '50%', 
                  bgcolor: color,
                  border: '1px solid #ddd'
                }} 
              />
            </ListItemIcon>
            <ListItemText>
              {color === '#FFFF00' ? 'Amarillo' : 
               color === '#FF9999' ? 'Rojo' :
               color === '#99FF99' ? 'Verde' :
               color === '#9999FF' ? 'Azul' :
               color === '#FF99FF' ? 'Rosa' : 'Color'}
            </ListItemText>
          </MenuItem>
        ))}
      </Menu>
      
      {/* Notificaciones */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default DocxPreviewPanel;