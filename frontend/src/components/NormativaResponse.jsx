import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css'; // Add this for code highlighting styles
import { BookOpen, AlertCircle, FileText, Copy, Check, Search, Info, Tag } from 'lucide-react';

const NormativaResponse = ({ responseData }) => {
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);
  const [activeTab, setActiveTab] = useState('content');
  const [expandedSections, setExpandedSections] = useState({});
  
  // Initialize expanded sections when data arrives
  useEffect(() => {
    if (responseData) {
      const initialExpandState = {};
      // Set all sections expanded by default
      setExpandedSections(initialExpandState);
    }
  }, [responseData]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(responseData.response).then(() => {
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    });
  };

  // Toggle section expansion
  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  if (!responseData) {
    return <div>No response data available</div>;
  }

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
      
      {/* Contenido principal de la respuesta con Markdown avanzado */}
      {activeTab === 'content' && (
        <div className="prose max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              // Títulos con estilos personalizados
              h1: ({node, ...props}) => <h1 className="text-2xl font-bold my-4 text-gray-800" {...props} />,
              h2: ({node, ...props}) => <h2 className="text-xl font-bold my-3 text-gray-800 border-b pb-1" {...props} />,
              h3: ({node, ...props}) => <h3 className="text-lg font-bold my-2 text-gray-800" {...props} />,
              h4: ({node, ...props}) => <h4 className="text-md font-bold my-2 text-gray-800" {...props} />,
              
              // Listas
              ul: ({node, ...props}) => <ul className="list-disc pl-5 my-2 space-y-1" {...props} />,
              ol: ({node, ...props}) => <ol className="list-decimal pl-5 my-2 space-y-1" {...props} />,
              li: ({node, ...props}) => <li className="my-1" {...props} />,
              
              // Enlaces
              a: ({node, ...props}) => <a className="text-blue-600 hover:text-blue-800 underline" {...props} />,
              
              // Elementos de texto
              p: ({node, ...props}) => <p className="my-2 text-gray-700 leading-relaxed" {...props} />,
              strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
              em: ({node, ...props}) => <em className="italic text-gray-800" {...props} />,
              
              // Bloques de código
              code: ({node, inline, className, children, ...props}) => {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <div className="my-4 rounded-md overflow-hidden">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </div>
                ) : (
                  <code className="px-1 py-0.5 bg-gray-100 rounded text-red-600 text-sm" {...props}>
                    {children}
                  </code>
                );
              },
              
              // Citas
              blockquote: ({node, ...props}) => (
                <blockquote className="border-l-4 border-gray-300 pl-4 py-1 my-3 text-gray-700 italic" {...props} />
              ),
              
              // Tablas
              table: ({node, ...props}) => (
                <div className="my-4 overflow-x-auto">
                  <table className="min-w-full border-collapse border border-gray-300" {...props} />
                </div>
              ),
              thead: ({node, ...props}) => <thead className="bg-gray-100" {...props} />,
              tbody: ({node, ...props}) => <tbody className="divide-y divide-gray-300" {...props} />,
              tr: ({node, ...props}) => <tr className="hover:bg-gray-50" {...props} />,
              th: ({node, ...props}) => <th className="border border-gray-300 px-4 py-2 text-left font-semibold" {...props} />,
              td: ({node, ...props}) => <td className="border border-gray-300 px-4 py-2 text-gray-700" {...props} />
            }}
          >
            {responseData.response}
          </ReactMarkdown>
        </div>
      )}
      
      {/* Vista de metadatos */}
      {activeTab === 'metadata' && (
        <div className="space-y-6">
          {responseData.metadata && Object.keys(responseData.metadata).length > 0 ? (
            <>
              {/* Fuentes de información */}
              {responseData.metadata.sources && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-3">
                    <FileText className="text-blue-600 mr-2" size={18} />
                    <h3 className="font-medium text-gray-800">Fuentes de información</h3>
                  </div>
                  <ul className="space-y-2">
                    {responseData.metadata.sources.map((source, index) => (
                      <li key={index} className="flex items-start">
                        <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-2 mt-0.5">
                          {source.type || 'Documento'}
                        </span>
                        <span className="text-gray-700">{source.title || source.id || `Fuente ${index + 1}`}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {/* Confianza y otras métricas */}
              {responseData.metadata.confidence !== undefined && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-3">
                    <Tag className="text-blue-600 mr-2" size={18} />
                    <h3 className="font-medium text-gray-800">Métricas de respuesta</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Confianza</p>
                      <div className="flex items-center">
                        <div className="w-full bg-gray-200 rounded-full h-2 mr-3">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${responseData.metadata.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-700">
                          {Math.round(responseData.metadata.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                    
                    {/* Otras métricas si están disponibles */}
                    {responseData.metadata.relevance !== undefined && (
                      <div>
                        <p className="text-sm text-gray-500 mb-1">Relevancia</p>
                        <div className="flex items-center">
                          <div className="w-full bg-gray-200 rounded-full h-2 mr-3">
                            <div 
                              className="bg-green-600 h-2 rounded-full" 
                              style={{ width: `${responseData.metadata.relevance * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-gray-700">
                            {Math.round(responseData.metadata.relevance * 100)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Otros metadatos disponibles */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center mb-3">
                  <Info className="text-blue-600 mr-2" size={18} />
                  <h3 className="font-medium text-gray-800">Información adicional</h3>
                </div>
                <div className="space-y-2">
                  {Object.entries(responseData.metadata)
                    .filter(([key]) => !['sources', 'confidence', 'relevance'].includes(key))
                    .map(([key, value]) => (
                      <div key={key} className="flex">
                        <span className="text-sm font-medium text-gray-600 min-w-[120px]">
                          {key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}:
                        </span>
                        <span className="text-sm text-gray-700 ml-2">
                          {typeof value === 'object' 
                            ? JSON.stringify(value) 
                            : String(value)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center p-6 bg-gray-50 rounded-lg">
              <AlertCircle className="text-gray-400 mr-2" size={20} />
              <p className="text-gray-500">No hay metadatos disponibles para esta respuesta.</p>
            </div>
          )}
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