// frontend/src/components/RichTextEditor.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Highlight from '@tiptap/extension-highlight';
import Placeholder from '@tiptap/extension-placeholder';
import { 
  Box, 
  Paper, 
  Divider, 
  Tooltip, 
  IconButton, 
  Button, 
  Stack
} from '@mui/material';
import FormatBoldIcon from '@mui/icons-material/FormatBold';
import FormatItalicIcon from '@mui/icons-material/FormatItalic';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';
import CodeIcon from '@mui/icons-material/Code';
import SaveIcon from '@mui/icons-material/Save';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import HighlightIcon from '@mui/icons-material/Highlight';

/**
 * Editor de texto enriquecido basado en TipTap
 * 
 * @param {Object} props - Props del componente
 * @param {string} props.content - Contenido inicial del editor
 * @param {Function} props.onChange - Función llamada cuando cambia el contenido
 * @param {string} props.placeholder - Texto de placeholder
 */
const RichTextEditor = ({ content = '', onChange, placeholder = 'Escriba aquí...' }) => {
  const [localContent, setLocalContent] = useState(content);
  
  // Configurar el editor
  const editor = useEditor({
    extensions: [
      StarterKit,
      Highlight,
      Placeholder.configure({
        placeholder: placeholder,
      }),
    ],
    content: content,
    onUpdate: ({ editor }) => {
      const html = editor.getHTML();
      setLocalContent(html);
    },
  });
  
  // Actualizar contenido cuando cambia la prop
  useEffect(() => {
    if (editor && content !== localContent) {
      editor.commands.setContent(content);
    }
  }, [content, editor]);
  
  // Manejar guardado
  const handleSave = useCallback(() => {
    if (onChange && localContent) {
      onChange(localContent);
    }
  }, [onChange, localContent]);
  
  if (!editor) {
    return null;
  }
  
  // Barra de herramientas personalizada
  const MenuBar = () => {
    return (
      <Box sx={{ 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: 0.5, 
        mb: 1.5, 
        bgcolor: 'background.default', 
        p: 1, 
        borderRadius: 1
      }}>
        <Tooltip title="Negrita">
          <IconButton
            onClick={() => editor.chain().focus().toggleBold().run()}
            className={editor.isActive('bold') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('bold') ? "primary" : "default"}
          >
            <FormatBoldIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Cursiva">
          <IconButton
            onClick={() => editor.chain().focus().toggleItalic().run()}
            className={editor.isActive('italic') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('italic') ? "primary" : "default"}
          >
            <FormatItalicIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Resaltar">
          <IconButton
            onClick={() => editor.chain().focus().toggleHighlight().run()}
            className={editor.isActive('highlight') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('highlight') ? "primary" : "default"}
          >
            <HighlightIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
        
        <Tooltip title="Lista con viñetas">
          <IconButton
            onClick={() => editor.chain().focus().toggleBulletList().run()}
            className={editor.isActive('bulletList') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('bulletList') ? "primary" : "default"}
          >
            <FormatListBulletedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Lista numerada">
          <IconButton
            onClick={() => editor.chain().focus().toggleOrderedList().run()}
            className={editor.isActive('orderedList') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('orderedList') ? "primary" : "default"}
          >
            <FormatListNumberedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Cita">
          <IconButton
            onClick={() => editor.chain().focus().toggleBlockquote().run()}
            className={editor.isActive('blockquote') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('blockquote') ? "primary" : "default"}
          >
            <FormatQuoteIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Código">
          <IconButton
            onClick={() => editor.chain().focus().toggleCodeBlock().run()}
            className={editor.isActive('codeBlock') ? 'is-active' : ''}
            size="small"
            color={editor.isActive('codeBlock') ? "primary" : "default"}
          >
            <CodeIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />
        
        <Tooltip title="Deshacer">
          <IconButton
            onClick={() => editor.chain().focus().undo().run()}
            disabled={!editor.can().undo()}
            size="small"
          >
            <UndoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Rehacer">
          <IconButton
            onClick={() => editor.chain().focus().redo().run()}
            disabled={!editor.can().redo()}
            size="small"
          >
            <RedoIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <Box sx={{ flexGrow: 1 }} />
        
        <Button
          variant="contained"
          size="small"
          startIcon={<SaveIcon />}
          onClick={handleSave}
          disabled={!localContent.trim()}
        >
          Guardar
        </Button>
      </Box>
    );
  };
  
  return (
    <Paper 
      variant="outlined" 
      sx={{ 
        overflow: 'hidden',
        '.ProseMirror': {
          p: 2,
          minHeight: '150px',
          maxHeight: '350px',
          overflowY: 'auto',
          '&:focus': {
            outline: 'none',
          },
          '& p.is-editor-empty:first-child::before': {
            color: 'text.disabled',
            content: 'attr(data-placeholder)',
            float: 'left',
            height: 0,
            pointerEvents: 'none',
          },
        },
      }}
    >
      <MenuBar />
      <EditorContent editor={editor} />
    </Paper>
  );
};

export default RichTextEditor;