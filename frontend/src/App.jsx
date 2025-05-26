// frontend/src/App.jsx (Modificado)
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import MainPage from './pages/MainPage';
import ConsultaHistorialPage from './pages/ConsultaHistorialPage';
import HistorialPage from './pages/HistorialPage';
import NovedadesPage from './pages/NovedadesPage';
import ThemeProvider from './theme/ThemeProvider';
import { useQueryStore } from './contexts/store';

// Importamos los componentes que necesitaremos para la previsualización
import DocxPreviewPanel from './components/DocxPreviewPanel';

function App() {
  // Get the function to load stored queries
  const loadStoredQueries = useQueryStore(state => state.loadStoredQueries);

  useEffect(() => {
    loadStoredQueries();
  }, [loadStoredQueries]);

  return (
    <ThemeProvider>
      <Router>
        <AppLayout>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/consulta/:id" element={<ConsultaHistorialPage />} />
            <Route path="/historial" element={<HistorialPage />} />
            <Route path="/novedades" element={<NovedadesPage />} />
          </Routes>
        </AppLayout>
      </Router>
    </ThemeProvider>
  );
}

export default App;