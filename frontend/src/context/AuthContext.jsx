import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiService } from '../services/api';

const AuthContext = createContext(null);

const TOKEN_KEY = 'nutrisense_token';
const USER_KEY = 'nutrisense_user';

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => {
    try {
      return localStorage.getItem(TOKEN_KEY) || null;
    } catch {
      return null;
    }
  });

  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem(USER_KEY);
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [isLoading, setIsLoading] = useState(true);

  // Validate stored token on mount
  useEffect(() => {
    async function verifyAuth() {
      if (!token) {
        setIsLoading(false);
        return;
      }

      try {
        const profile = await apiService.getMe();
        if (profile && profile.user_id) {
          setUser(profile);
          localStorage.setItem(USER_KEY, JSON.stringify(profile));
        } else {
          throw new Error('Invalid profile');
        }
      } catch (err) {
        // Token expired, revoked, or server unavailable
        console.warn('Session check failed or expired:', err.message);
        // Only clear if explicitly unauthorized 401
        if (err.status === 401) {
          setToken(null);
          setUser(null);
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
        }
      } finally {
        setIsLoading(false);
      }
    }

    verifyAuth();
  }, [token]);

  const login = async (email, password) => {
    const res = await apiService.login({ email, password });
    if (res && res.access_token) {
      setToken(res.access_token);
      setUser(res.user);
      localStorage.setItem(TOKEN_KEY, res.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(res.user));
      return res.user;
    }
    throw new Error('Login failed: Invalid server response.');
  };

  const register = async ({ name, email, password, confirm_password, role = 'health_worker' }) => {
    // 1. Create account
    const regRes = await apiService.register({
      name,
      email,
      password,
      confirm_password,
      role
    });

    // 2. Automatically log in to obtain JWT token
    if (regRes && regRes.user_id) {
      return await login(email, password);
    }
    return regRes;
  };

  const logout = async () => {
    try {
      if (token) {
        await apiService.logout();
      }
    } catch (err) {
      console.warn('Server logout notice:', err.message);
    } finally {
      setToken(null);
      setUser(null);
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  };

  const googleLogin = async (payload = {}) => {
    const res = await apiService.googleLogin(payload);
    if (res && res.access_token) {
      setToken(res.access_token);
      setUser(res.user);
      localStorage.setItem(TOKEN_KEY, res.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(res.user));
      return res.user;
    }
    throw new Error('Google sign-in failed: Invalid server response.');
  };

  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState('signin'); // 'signin' | 'signup'

  const openAuthModal = (mode = 'signin') => {
    setAuthModalMode(mode);
    setIsAuthModalOpen(true);
  };

  const closeAuthModal = () => {
    setIsAuthModalOpen(false);
  };

  const value = {
    user,
    token,
    isAuthenticated: Boolean(token && user),
    isLoading,
    login,
    googleLogin,
    register,
    logout,
    isAuthModalOpen,
    authModalMode,
    setAuthModalMode,
    openAuthModal,
    closeAuthModal
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
