import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import AuthModal from './components/AuthModal';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import ScreeningPage from './pages/ScreeningPage';
import ResultsPage from './pages/ResultsPage';
import AboutPage from './pages/AboutPage';
import SystemPage from './pages/SystemPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

export default function App() {
  const [screeningResult, setScreeningResult] = useState(null);

  const handleScreeningSuccess = (result) => {
    setScreeningResult(result);
  };

  return (
    <AuthProvider>
      <Router>
        <div className="app-container">
          <AuthModal />
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/screen" element={<ScreeningPage onScreeningSuccess={handleScreeningSuccess} />} />
              <Route path="/results" element={<ResultsPage screeningResult={screeningResult} />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/system" element={<SystemPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/signin" element={<LoginPage />} />
              <Route path="/signup" element={<RegisterPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </AuthProvider>
  );
}
