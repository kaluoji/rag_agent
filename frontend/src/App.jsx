// frontend/src/App.jsx
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import MainPage from './pages/MainPage';
import ConsultaHistorialPage from './pages/ConsultaHistorialPage';
import NovedadesPage from './pages/NovedadesPage';
import ThemeProvider from './theme/ThemeProvider';
import { useQueryStore } from './contexts/store';

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
            <Route path="/novedades" element={<NovedadesPage />} />
            {/* Puedes agregar más rutas aquí según sea necesario */}
          </Routes>
        </AppLayout>
      </Router>
    </ThemeProvider>
  );
}

export default App;