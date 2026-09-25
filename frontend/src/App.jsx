import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import ScreeningPage from './pages/ScreeningPage';
import ResultsPage from './pages/ResultsPage';
import AboutPage from './pages/AboutPage';
import SystemPage from './pages/SystemPage';

export default function App() {
  const [screeningResult, setScreeningResult] = useState(null);

  const handleScreeningSuccess = (result) => {
    setScreeningResult(result);
  };

  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route
              path="/screen"
              element={<ScreeningPage onScreeningSuccess={handleScreeningSuccess} />}
            />
            <Route
              path="/results"
              element={<ResultsPage screeningResult={screeningResult} />}
            />
            <Route path="/about" element={<AboutPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}
