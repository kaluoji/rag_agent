import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AppLayout from './layouts/AppLayout';
import MainPage from './pages/MainPage';

function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<MainPage />} />
          {/* Puedes agregar más rutas aquí según sea necesario */}
        </Routes>
      </AppLayout>
    </Router>
  );
}

export default App;