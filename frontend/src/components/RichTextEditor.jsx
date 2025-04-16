import React, { useState } from 'react';
import { Box, TextField, Button } from '@mui/material';

const RichTextEditor = ({ content = '', onChange, placeholder }) => {
  const [editorContent, setEditorContent] = useState(content);
  
  const handleChange = (e) => {
    const newContent = e.target.value;
    setEditorContent(newContent);
  };
  
  const handleSave = () => {
    if (onChange) {
      onChange(editorContent);
    }
  };
  
  return (
    <Box>
      <TextField
        fullWidth
        multiline
        rows={4}
        variant="outlined"
        value={editorContent}
        onChange={handleChange}
        placeholder={placeholder || 'Escriba su texto aquí...'}
        sx={{ mb: 2 }}
      />
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleSave}
      >
        Guardar
      </Button>
    </Box>
  );
};

export default RichTextEditor;