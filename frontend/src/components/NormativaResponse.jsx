import React, { useState, useEffect } from 'react';
import { BookOpen, AlertCircle, FileText, Copy, Check, Search, Info, Tag } from 'lucide-react';

const NormativaResponse = () => {
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);
  const [activeTab, setActiveTab] = useState('content');
  const [expandedSections, setExpandedSections] = useState({});
  
  // Esta función procesará el contenido de la respuesta para detectar estructura
  // y aplicar formato adecuado
  const processContent = (content) => {
    if (!content) return [];
    
    // Dividir por saltos de línea y procesar cada línea
    const lines = content.split('\n');
    const processed = [];
    let currentSection = null;
    
    lines.forEach((line, index) => {
      const trimmedLine = line.trim();
      
      // Detectar títulos de sección (##, ###, o numerados como 1., 2., etc.)
      if (trimmedLine.startsWith('##') || /^(\d+\.|\*|\-)\s/.test(trimmedLine)) {
        // Es un título o elemento de lista de primer nivel
        if (currentSection) {
          processed.push(currentSection);
        }
        
        currentSection = {
          id: `section-${index}`,
          title: trimmedLine.replace(/^(##|\d+\.|\*|\-)\s/, ''),
          type: trimmedLine.startsWith('##') ? 'heading' : 'list',
          content: [],
          level: trimmedLine.startsWith('##') ? (trimmedLine.startsWith('###') ? 3 : 2) : 1
        };
      } else if (trimmedLine.startsWith('-') || trimmedLine.startsWith('*')) {
        // Es un elemento de sublista
        if (currentSection) {
          currentSection.content.push({
            type: 'list-item',
            text: trimmedLine.replace(/^(\-|\*)\s/, '')
          });
        } else {
          // Si no hay sección activa, crear una nueva con este elemento
          currentSection = {
            id: `section-${index}`,
            title: 'Lista',
            type: 'list',
            content: [{
              type: 'list-item',
              text: trimmedLine.replace(/^(\-|\*)\s/, '')
            }],
            level: 1
          };
        }
      } else if (trimmedLine === '') {
        // Línea vacía, posiblemente separador entre párrafos
        if (currentSection && currentSection.content.length > 0 && 
            currentSection.content[currentSection.content.length - 1].type === 'paragraph') {
          // Agregar salto dentro del párrafo existente
          const lastPara = currentSection.content[currentSection.content.length - 1];
          lastPara.text += '\n\n';
        }
      } else {
        // Es contenido de texto normal
        if (currentSection) {
          // Si la última entrada es un párrafo, agregar a él
          if (currentSection.content.length > 0 && 
              currentSection.content[currentSection.content.length - 1].type === 'paragraph') {
            const lastPara = currentSection.content[currentSection.content.length - 1];
            lastPara.text += ' ' + trimmedLine;
          } else {
            // Si no, crear nuevo párrafo
            currentSection.content.push({
              type: 'paragraph',
              text: trimmedLine
            });
          }
        } else {
          // Si no hay sección activa, crear una nueva con este texto
          currentSection = {
            id: `section-${index}`,
            title: 'Introducción',
            type: 'text',
            content: [{
              type: 'paragraph',
              text: trimmedLine
            }],
            level: 1
          };
        }
      }
    });
    
    // Agregar la última sección si existe
    if (currentSection) {
      processed.push(currentSection);
    }
    
    return processed;
  };

  // Ejemplo de respuesta que podría venir del backend
  // Esto sería reemplazado por los datos reales de tu API
  const responseData = {
    query: "Cuales son los requisitos para cumplir con PCI DSS para comercios electrónicos?",
    response: `Los requisitos para cumplir con PCI DSS (Payment Card Industry Data Security Standard) para comercios electrónicos son fundamentales para garantizar la seguridad en el manejo de datos de tarjetas de pago. A continuación se presentan los requisitos esenciales desglosados en varias categorías:

## 1. Construir y mantener una red segura
- Instalar y mantener un firewall para proteger los datos del titular de la tarjeta.
- No utilizar las contraseñas predeterminadas proporcionadas por los proveedores de sistemas.

## 2. Proteger los datos del titular de la tarjeta
- Proteger los datos almacenados del titular de la tarjeta mediante cifrado.
- Cifrar la transmisión de los datos del titular de la tarjeta a través de redes públicas abiertas.

## 3. Mantener un programa de gestión de vulnerabilidades
- Usar y actualizar regularmente software o programas antivirus.
- Desarrollar y mantener sistemas y aplicaciones seguras.

## 4. Implementar medidas de control de acceso
- Restringir el acceso a los datos según la necesidad de conocer.
- Asignar un ID único a cada persona con acceso a equipos.
- Restringir el acceso físico a los datos del titular de la tarjeta.

## 5. Monitorear y probar las redes regularmente
- Rastrear y monitorear todos los accesos a los recursos de la red y datos del titular.
- Probar regularmente los sistemas y procesos de seguridad.

## 6. Mantener una política de seguridad de la información
- Mantener una política que aborde la seguridad de la información para todo el personal.

Para comercios electrónicos, es especialmente importante implementar medidas de seguridad adicionales como autenticación multifactor para transacciones en línea y revisiones de código de aplicaciones web para detectar vulnerabilidades.`,
    metadata: {
      processingTime: "11.52s",
      sourcesConsulted: [
        "Reglas Básicas de Visa: Seguridad de la Información",
        "Visa NCA Guidelines and Reporting Procedures",
        "Actualizaciones en el documento 'What To Do If Compromised'"
      ],
      confidence: 0.92,
      relevanceScore: 0.95,
      tags: ["PCI DSS", "Comercio electrónico", "Seguridad de datos", "Tarjetas de pago"]
    }
  };

  // Procesar la respuesta para mostrarla estructurada
  const processedSections = processContent(responseData.response);
  
  // Función para alternar la expansión de secciones
  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  // Para el componente real, esto se ejecutaría cuando lleguen los datos
  useEffect(() => {
    // Inicializar todas las secciones como expandidas por defecto
    const initialExpandState = {};
    processedSections.forEach(section => {
      initialExpandState[section.id] = true;
    });
    setExpandedSections(initialExpandState);
  }, []);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(responseData.response).then(() => {
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    });
  };

  // Renderizar contenido basado en el tipo
  const renderContent = (content) => {
    if (!content) return null;
    
    return content.map((item, index) => {
      if (item.type === 'paragraph') {
        return (
          <p key={index} className="text-gray-700 mb-4">
            {item.text.split('\n\n').map((paragraph, i) => (
              <React.Fragment key={i}>
                {paragraph}
                {i < item.text.split('\n\n').length - 1 && <br /><br />}
              </React.Fragment>
            ))}
          </p>
        );
      } else if (item.type === 'list-item') {
        return (
          <li key={index} className="flex items-start mb-2">
            <div className="text-blue-600 mr-2 mt-1 flex-shrink-0">•</div>
            <span className="text-gray-700">{item.text}</span>
          </li>
        );
      }
      return null;
    });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-4xl mx-auto">
      {/* Cabecera con la consulta */}
      <div className="mb-6">
        <div className="flex items-center mb-4 bg-gray-50 p-4 rounded-lg">
          <Search className="text-blue-600 mr-3" size={20} />
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-1">Consulta</h3>
            <p className="text-lg font-semibold text-gray-800">{responseData.query}</p>
          </div>
        </div>
      </div>
      
      {/* Tabs para cambiar entre contenido y metadatos */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('content')}
            className={`mr-8 py-4 px-1 font-medium text-sm ${
              activeTab === 'content' 
                ? 'border-b-2 border-blue-500 text-blue-600' 
                : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <BookOpen className="inline-block mr-2" size={16} />
            Contenido
          </button>
          <button
            onClick={() => setActiveTab('metadata')}
            className={`py-4 px-1 font-medium text-sm ${
              activeTab === 'metadata' 
                ? 'border-b-2 border-blue-500 text-blue-600' 
                : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Info className="inline-block mr-2" size={16} />
            Información
          </button>
        </nav>
      </div>
      
      {/* Contenido principal de la respuesta */}
      {activeTab === 'content' && (
        <div className="space-y-6 mb-6">
          {processedSections.map((section, index) => (
            <div key={section.id} className="border rounded-lg overflow-hidden">
              {section.type === 'text' ? (
                <div className="p-5">
                  {renderContent(section.content)}
                </div>
              ) : (
                <>
                  <div 
                    className={`flex justify-between items-center p-4 cursor-pointer ${
                      expandedSections[section.id] ? 'bg-blue-50' : 'bg-gray-50'
                    }`}
                    onClick={() => toggleSection(section.id)}
                  >
                    <h3 className={`font-semibold ${
                      section.level <= 2 ? 'text-lg text-gray-800' : 'text-md text-gray-700'
                    }`}>
                      {section.title}
                    </h3>
                    <span className="text-gray-500">
                      {expandedSections[section.id] ? (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M5 10a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1z" clipRule="evenodd" />
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                        </svg>
                      )}
                    </span>
                  </div>
                  
                  {expandedSections[section.id] && (
                    <div className="p-5 bg-white border-t">
                      {section.type === 'list' ? (
                        <ul className="space-y-2">
                          {renderContent(section.content)}
                        </ul>
                      ) : (
                        renderContent(section.content)
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Vista de metadatos */}
      {activeTab === 'metadata' && (
        <div className="space-y-6">
          {/* Tiempo de procesamiento */}
          <div className="p-4 border rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2 flex items-center">
              <svg className="mr-2 h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Tiempo de procesamiento
            </h3>
            <p className="text-gray-700">{responseData.metadata.processingTime}</p>
          </div>
          
          {/* Fuentes consultadas */}
          <div className="p-4 border rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2 flex items-center">
              <FileText className="mr-2 text-gray-500" size={18} />
              Fuentes consultadas
            </h3>
            <ul className="space-y-1">
              {responseData.metadata.sourcesConsulted.map((source, idx) => (
                <li key={idx} className="text-gray-700 text-sm flex items-center">
                  <span className="text-blue-600 mr-2">•</span>
                  {source}
                </li>
              ))}
            </ul>
          </div>
          
          {/* Métricas de calidad */}
          <div className="p-4 border rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2 flex items-center">
              <AlertCircle className="mr-2 text-gray-500" size={18} />
              Métricas de calidad
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Confianza</p>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${responseData.metadata.confidence * 100}%` }}
                  ></div>
                </div>
                <p className="text-sm text-right text-gray-700 mt-1">
                  {Math.round(responseData.metadata.confidence * 100)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Relevancia</p>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${responseData.metadata.relevanceScore * 100}%` }}
                  ></div>
                </div>
                <p className="text-sm text-right text-gray-700 mt-1">
                  {Math.round(responseData.metadata.relevanceScore * 100)}%
                </p>
              </div>
            </div>
          </div>
          
          {/* Etiquetas */}
          <div className="p-4 border rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-2 flex items-center">
              <Tag className="mr-2 text-gray-500" size={18} />
              Etiquetas
            </h3>
            <div className="flex flex-wrap gap-2">
              {responseData.metadata.tags.map((tag, idx) => (
                <span key={idx} className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Acciones */}
      <div className="flex justify-end mt-6 pt-4 border-t">
        <button 
          onClick={copyToClipboard}
          className="flex items-center bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          {copiedToClipboard ? (
            <>
              <Check className="mr-2" size={16} />
              Copiado
            </>
          ) : (
            <>
              <Copy className="mr-2" size={16} />
              Copiar resultado
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default NormativaResponse;