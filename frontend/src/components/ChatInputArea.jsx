import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  IconButton,
  InputBase,
  Paper,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const ChatInputArea = ({ onSendMessage, isLoading }) => {
  const [message, setMessage] = useState('');
  const [focused, setFocused] = useState(false);
  const textAreaRef = useRef(null);
  const formRef = useRef(null);

  // Función para ajustar automáticamente la altura del textarea
  const adjustTextAreaHeight = () => {
    const textArea = textAreaRef.current;
    if (!textArea) return;

    // Restablecer altura para recalcular
    textArea.style.height = 'auto';
    
    // Establecer la altura basada en el contenido (scrollHeight)
    const newHeight = Math.min(textArea.scrollHeight, 200); // máximo 200px
    textArea.style.height = `${newHeight}px`;
  };

  // Ajustar altura cuando cambia el mensaje
  useEffect(() => {
    adjustTextAreaHeight();
  }, [message]);

  const handleChange = (e) => {
    setMessage(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;
    
    onSendMessage(message);
    setMessage('');
  };

  // Manejar envío con Enter (pero permitir nueva línea con Shift+Enter)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <Box 
      sx={{ 
        p: '0 16px 16px 16px',
        position: 'relative'
      }}
    >
      <form ref={formRef} onSubmit={handleSubmit}>
        <Paper
          elevation={0}
          sx={{
            display: 'flex',
            alignItems: 'flex-end',
            borderRadius: '24px',
            p: '4px 4px 4px 16px',
            backgroundColor: '#fff',
            border: '1px solid rgba(0, 0, 0, 0.12)',
            boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)',
            transition: 'all 0.3s ease',
            '&:hover': {
              border: '1px solid rgba(0, 0, 0, 0.23)',
              boxShadow: '0 3px 8px rgba(0, 0, 0, 0.07)'
            },
            '&:focus-within': {
              border: '1px solid rgba(79, 6, 42, 0.5)',
              boxShadow: '0 3px 10px rgba(79, 6, 42, 0.12)',
            }
          }}
        >
          <InputBase
            inputRef={textAreaRef}
            fullWidth
            multiline
            placeholder="Escriba su consulta..."
            value={message}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={isLoading}
            inputProps={{
              'aria-label': 'consulta',
              style: {
                padding: '8px 0',
                lineHeight: '1.5',
                minHeight: '24px',
                scrollbarWidth: 'thin',
                scrollbarColor: 'rgba(79, 6, 42, 0.2) transparent'
              }
            }}
            sx={{
              flex: 1,
              fontSize: '0.95rem',
              '& .MuiInputBase-input': {
                '&::-webkit-scrollbar': {
                  width: '6px',
                },
                '&::-webkit-scrollbar-track': {
                  background: 'transparent',
                  borderRadius: '10px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: 'rgba(79, 6, 42, 0.2)',
                  borderRadius: '10px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: 'rgba(79, 6, 42, 0.4)',
                }
              }
            }}
          />
          <Tooltip title="Enviar consulta" arrow placement="top">
            <span>
              <IconButton 
                color="primary" 
                type="submit" 
                className="send-button"
                disabled={isLoading || !message.trim()}
                sx={{ 
                  p: '8px',
                  m: '4px',
                  backgroundColor: '#4F062A',
                  color: 'white',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: '#38071C',
                    transform: 'scale(1.05)'
                  },
                  '&.Mui-disabled': {
                    backgroundColor: 'action.disabledBackground',
                    color: 'action.disabled'
                  }
                }}
              >
                <SendIcon />
              </IconButton>
            </span>
          </Tooltip>
        </Paper>
        <Box 
          sx={{ 
            textAlign: 'center', 
            mt: 1, 
            fontSize: '0.75rem', 
            color: 'text.secondary', 
            opacity: 0.7 
          }}
        >
          Pulse Enter para enviar, Shift+Enter para nueva línea
        </Box>
      </form>
    </Box>
  );
};

export default ChatInputArea;