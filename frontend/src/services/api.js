/**
 * NutriSense AI Backend API Service Layer.
 * Communicates strictly with Step 18 FastAPI endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  constructor(message, status, code = 'API_ERROR', details = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const token = typeof window !== 'undefined' ? localStorage.getItem('nutrisense_token') : null;
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {})
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers
    });

    let data;
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = { message: await response.text() };
    }

    if (!response.ok) {
      const err = data.error || {};
      const message = err.message || data.message || `Request failed with status ${response.status}`;
      const code = err.code || `HTTP_${response.status}`;
      const details = err.details || [];
      throw new ApiError(message, response.status, code, details);
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Network failure / server offline
    throw new ApiError(
      'Unable to connect to the NutriSense AI backend. Please verify that the FastAPI server is running on http://127.0.0.1:8000.',
      0,
      'NETWORK_ERROR',
      [error.message]
    );
  }
}

export const apiService = {
  /**
   * Evaluates child undernutrition risk under Scenario A.
   * @param {Object} childData - Approved 30 Scenario-A features
   */
  async screenChild(childData) {
    return request('/api/v1/screen', {
      method: 'POST',
      body: JSON.stringify(childData)
    });
  },

  /**
   * Service liveness check.
   */
  async getHealth() {
    return request('/health');
  },

  /**
   * Model readiness and integrity check.
   */
  async getModelHealth() {
    return request('/api/v1/health/model');
  },

  /**
   * Safe public application and model metadata.
   */
  async getMetadata() {
    return request('/api/v1/metadata');
  },

  /**
   * Database connectivity and status check.
   */
  async getDatabaseHealth() {
    return request('/api/v1/health/database');
  },

  /**
   * Retrieves recent screening history records from MongoDB.
   */
  async getScreenings(limit = 20, skip = 0) {
    return request(`/api/v1/screenings?limit=${limit}&skip=${skip}`);
  },

  /**
   * Retrieves a single screening record by ID from MongoDB.
   */
  async getScreeningById(screeningId) {
    return request(`/api/v1/screenings/${screeningId}`);
  },

  /**
   * Deletes a screening record by ID from MongoDB.
   */
  async deleteScreening(screeningId) {
    return request(`/api/v1/screenings/${screeningId}`, {
      method: 'DELETE'
    });
  },

  // --------------------------------------------------------------------------
  // Authentication & Health Worker Account Services (Phases 1-7)
  // --------------------------------------------------------------------------

  /**
   * Registers a new health worker account.
   * @param {Object} userData - { name, email, password, confirm_password, role }
   */
  async register(userData) {
    return request('/api/v1/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData)
    });
  },

  /**
   * Authenticates health worker and returns JWT bearer token.
   * @param {Object} credentials - { email, password }
   */
  async login(credentials) {
    return request('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials)
    });
  },

  /**
   * Authenticates user via Google SSO.
   * @param {Object} payload - { token, email, name }
   */
  async googleLogin(payload = {}) {
    return request('/api/v1/auth/google', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  },

  /**
   * Retrieves authenticated health worker profile via JWT Bearer token.
   */
  async getMe() {
    return request('/api/v1/auth/me');
  },

  /**
   * Logs out user and revokes current JWT token on server.
   */
  async logout() {
    return request('/api/v1/auth/logout', {
      method: 'POST'
    });
  }
};
