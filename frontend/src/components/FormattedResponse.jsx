// frontend/src/components/FormattedResponse.jsx
import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Chip, 
  Divider, 
  List, 
  ListItem, 
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  useTheme
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ArticleIcon from '@mui/icons-material/Article';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoIcon from '@mui/icons-material/Info';
import SecurityIcon from '@mui/icons-material/Security';
import GavelIcon from '@mui/icons-material/Gavel';
import NumbersIcon from '@mui/icons-material/Numbers';

/**
 * Componente para mostrar respuestas de consultas normativas con formato mejorado
 * 
 * @param {Object} props - Props del componente
 * @param {string} props.response - Texto de respuesta de la consulta
 * @param {string} props.query - Texto de la consulta original
 * @param {Object} props.metadata - Metadatos adicionales de la respuesta
 */
const FormattedResponse = ({ response, query, metadata = {} }) => {
  const theme = useTheme();
  const [formattedSections, setFormattedSections] = useState([]);
  const [regulationType, setRegulationType] = useState('');
  
  useEffect(() => {
    if (!response) return;
    
    // Detectar tipo de regulación
    const detectRegulationType = () => {
      const lowerResponse = response.toLowerCase();
      if (lowerResponse.includes('visa') && lowerResponse.includes('mastercard')) {
        return 'VISA y Mastercard';
      } else if (lowerResponse.includes('visa')) {
        return 'VISA';
      } else if (lowerResponse.includes('mastercard')) {
        return 'Mastercard';
      } else if (lowerResponse.includes('pci dss')) {
        return 'PCI DSS';
      } else if (lowerResponse.includes('gdpr') || lowerResponse.includes('rgpd')) {
        return 'GDPR/RGPD';
      } else {
        return 'Normativa';
      }
    };
    
    // Función para parsear y estructurar la respuesta
    const parseResponse = (text) => {
      // Dividir por secciones (usando diferentes heurísticas)
      let sections = [];
      
      // Verificar si hay patrones de secciones numeradas como "1." o títulos con ###
      if (text.match(/^#+\s+.+/m) || text.match(/^\d+\.\s+.+/m)) {
        // Separar por patrones de Markdown o numeración
        const lines = text.split('\n');
        let currentSection = { title: '', content: '', type: 'text' };
        
        lines.forEach(line => {
          // Detectar encabezados al estilo Markdown (###)
          if (line.match(/^#+\s+.+/)) {
            // Guardar la sección anterior si existe
            if (currentSection.content) {
              sections.push(currentSection);
            }
            
            // Iniciar nueva sección
            const titleText = line.replace(/^#+\s+/, '');
            currentSection = { 
              title: titleText, 
              content: '', 
              type: detectSectionType(titleText, '') 
            };
          } 
          // Detectar encabezados numerados (1., 2., etc.)
          else if (line.match(/^\d+\.\s+.+/) && !currentSection.content) {
            // Guardar la sección anterior si existe
            if (currentSection.content) {
              sections.push(currentSection);
            }
            
            // Iniciar nueva sección
            const titleText = line;
            currentSection = { 
              title: titleText, 
              content: '', 
              type: detectSectionType(titleText, '') 
            };
          } 
          // Detectar subítems con - o *
          else if (line.match(/^[-*]\s+.+/)) {
            currentSection.content += line + '\n';
            // Marcar como lista si no se ha hecho antes
            if (currentSection.type === 'text') {
              currentSection.type = 'list';
            }
          }
          // Contenido regular
          else {
            currentSection.content += line + '\n';
          }
        });
        
        // Añadir la última sección
        if (currentSection.content) {
          sections.push(currentSection);
        }
      } 
      // Si no hay un formato claro, dividir por párrafos
      else {
        const paragraphs = text.split(/\n\s*\n/);
        
        sections = paragraphs.map(paragraph => {
          // Detectar si es un párrafo de requisito, definición, etc.
          const type = detectSectionType('', paragraph);
          
          return {
            title: '',
            content: paragraph.trim(),
            type
          };
        });
      }
      
      // Procesar el contenido de cada sección para detectar listas
      sections = sections.map(section => {
        // Si ya es una lista, procesar los items
        if (section.type === 'list') {
          const lines = section.content.split('\n');
          const items = lines
            .filter(line => line.match(/^[-*]\s+.+/))
            .map(line => line.replace(/^[-*]\s+/, ''));
          
          return {
            ...section,
            items
          };
        }
        // Verificar si el contenido tiene patrones de lista numérica
        else if (section.content.match(/^\d+[.)]\s+.+/m)) {
          const lines = section.content.split('\n');
          const items = [];
          let currentItem = '';
          
          lines.forEach(line => {
            if (line.match(/^\d+[.)]\s+.+/)) {
              if (currentItem) {
                items.push(currentItem.trim());
              }
              currentItem = line.replace(/^\d+[.)]\s+/, '');
            } else if (line.trim()) {
              currentItem += ' ' + line.trim();
            }
          });
          
          if (currentItem) {
            items.push(currentItem.trim());
          }
          
          return {
            ...section,
            type: 'numbered_list',
            items
          };
        }
        
        return section;
      });
      
      return sections;
    };
    
    // Detectar el tipo de sección basado en su contenido
    const detectSectionType = (title, content) => {
      const lowerTitle = title.toLowerCase();
      const lowerContent = content.toLowerCase();
      const combined = lowerTitle + ' ' + lowerContent;
      
      if (combined.includes('requisito') || combined.includes('requirement') || 
          combined.match(/debe[n]?(\s+ser)?/)) {
        return 'requirement';
      }
      else if (combined.includes('definición') || combined.includes('definition') ||
              combined.includes('se define') || combined.includes('se refiere a')) {
        return 'definition';
      }
      else if (combined.includes('advertencia') || combined.includes('warning') ||
              combined.includes('precaución') || combined.includes('caution') ||
              combined.includes('no debe')) {
        return 'warning';
      }
      else if (combined.includes('seguridad') || combined.includes('security') ||
              combined.includes('protección') || combined.includes('protection')) {
        return 'security';
      }
      else if (combined.match(/artículo \d+/) || combined.match(/sección \d+/) ||
              combined.match(/article \d+/) || combined.match(/section \d+/)) {
        return 'article';
      }
      else if (combined.match(/\d+\.\d+/) || combined.match(/\d+\.\d+\.\d+/)) {
        return 'numbered';
      }
      else if (combined.includes('importante') || combined.includes('important') ||
              combined.includes('nota:') || combined.includes('note:')) {
        return 'important';
      }
      else if (combined.match(/-\s+/) || combined.match(/•\s+/) || 
               combined.match(/\*\s+/) || combined.match(/\d+\.\s+/)) {
        return 'list';
      }
      
      return 'text';
    };
    
    // Procesar la respuesta
    const regType = detectRegulationType();
    setRegulationType(regType);
    
    const sections = parseResponse(response);
    setFormattedSections(sections);
  }, [response]);
  
  // Obtener icono según el tipo de sección
  const getSectionIcon = (type) => {
    switch (type) {
      case 'requirement':
        return <CheckCircleOutlineIcon />;
      case 'warning':
        return <ErrorOutlineIcon />;
      case 'definition':
        return <InfoIcon />;
      case 'security':
        return <SecurityIcon />;
      case 'article':
        return <GavelIcon />;
      case 'numbered':
        return <NumbersIcon />;
      case 'important':
        return <InfoIcon color="primary" />;
      default:
        return <ArticleIcon />;
    }
  };
  
  // Obtener color según el tipo de sección
  const getSectionColor = (type) => {
    switch (type) {
      case 'requirement':
        return theme.palette.success.light;
      case 'warning':
        return theme.palette.warning.light;
      case 'definition':
        return theme.palette.info.light;
      case 'security':
        return theme.palette.secondary.light;
      case 'article':
        return theme.palette.primary.light;
      case 'important':
        return theme.palette.info.main;
      default:
        return 'transparent';
    }
  };
  
  // Renderizar sección según su tipo
  const renderSection = (section, index) => {
    const borderColor = getSectionColor(section.type);
    
    // Renderizar sección de tipo lista
    if (section.type === 'list' || section.type === 'numbered_list') {
      return (
        <Box 
          key={index} 
          sx={{ 
            mb: 3, 
            borderLeft: `4px solid ${borderColor}`,
            pl: 2,
            py: 1
          }}
        >
          {section.title && (
            <Typography variant="h6" gutterBottom>
              {section.title}
            </Typography>
          )}
          
          <List dense={false}>
            {section.items?.map((item, itemIndex) => (
              <ListItem key={itemIndex}>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  {section.type === 'numbered_list' ? 
                    <Typography variant="body1" fontWeight="bold">{itemIndex + 1}.</Typography> :
                    <CheckCircleOutlineIcon fontSize="small" />
                  }
                </ListItemIcon>
                <ListItemText primary={item} />
              </ListItem>
            ))}
          </List>
        </Box>
      );
    }
    
    // Renderizar sección de texto normal
    return (
      <Box 
        key={index} 
        sx={{ 
          mb: 3, 
          borderLeft: section.type !== 'text' ? `4px solid ${borderColor}` : 'none',
          pl: section.type !== 'text' ? 2 : 0,
          py: section.type !== 'text' ? 1 : 0
        }}
      >
        {section.title && (
          <Typography variant="h6" gutterBottom>
            {section.title}
          </Typography>
        )}
        
        <Typography variant="body1" paragraph>
          {section.content}
        </Typography>
      </Box>
    );
  };
  
  if (!response) {
    return null;
  }
  
  return (
    <Paper elevation={0} sx={{ p: 0 }}>
      {/* Encabezado */}
      <Box 
        sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mb: 2
        }}
      >
        <Typography variant="h6" color="primary">
          Respuesta sobre {regulationType}
        </Typography>
        
        <Chip 
          label={regulationType}
          color="primary"
          size="small"
          icon={<ArticleIcon />}
        />
      </Box>
      
      {/* Consulta original */}
      <Paper 
        variant="outlined" 
        sx={{ p: 2, mb: 3, bgcolor: theme.palette.background.default }}
      >
        <Typography variant="subtitle2" color="text.secondary">
          Consulta:
        </Typography>
        <Typography variant="body1">
          {query}
        </Typography>
      </Paper>
      
      {/* Respuesta formateada */}
      <Box sx={{ mb: 3 }}>
        {/* Si tenemos secciones procesadas */}
        {formattedSections.length > 0 ? (
          <Box>
            {formattedSections.map((section, index) => (
              renderSection(section, index)
            ))}
          </Box>
        ) : (
          /* Respuesta sin procesar como fallback */
          <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
            {response}
          </Typography>
        )}
      </Box>
      
      {/* Metadatos - Información adicional colapsable */}
      {Object.keys(metadata).length > 0 && (
        <Accordion sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Información adicional</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              {metadata.agent_used && (
                <Typography variant="body2" gutterBottom>
                  <strong>Agente utilizado:</strong> {metadata.agent_used}
                </Typography>
              )}
              
              {metadata.query_info && (
                <Box mt={1}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Categoría de consulta:</strong> {metadata.query_info.category || 'General'}
                  </Typography>
                  {metadata.query_info.confidence && (
                    <Typography variant="body2" gutterBottom>
                      <strong>Nivel de confianza:</strong> {(metadata.query_info.confidence * 100).toFixed(0)}%
                    </Typography>
                  )}
                </Box>
              )}
              
              {metadata.additional_info && (
                <Box mt={1}>
                  <Typography variant="subtitle2" gutterBottom>
                    Fuentes:
                  </Typography>
                  {Array.isArray(metadata.additional_info.sources) ? (
                    <List dense>
                      {metadata.additional_info.sources.map((source, idx) => (
                        <ListItem key={idx}>
                          <ListItemText primary={source} />
                        </ListItem>
                      ))}
                    </List>
                  ) : (
                    <Typography variant="body2">
                      {JSON.stringify(metadata.additional_info)}
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}
    </Paper>
  );
};

export default FormattedResponse;