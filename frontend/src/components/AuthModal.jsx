import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, X, Check, ArrowLeft, ChevronRight, User, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function AuthModal() {
  const {
    isAuthModalOpen,
    authModalMode,
    setAuthModalMode,
    closeAuthModal,
    login,
    register,
    googleLogin
  } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showGoogleChooser, setShowGoogleChooser] = useState(false);
  const [customGoogleEmail, setCustomGoogleEmail] = useState('');
  const [showCustomGoogleInput, setShowCustomGoogleInput] = useState(false);

  const isSignUp = authModalMode === 'signup';

  // Reset state whenever modal opens or mode changes
  useEffect(() => {
    if (isAuthModalOpen) {
      setEmail('');
      setPassword('');
      setConfirm('');
      setShowPassword(false);
      setShowConfirm(false);
      setError('');
      setShowGoogleChooser(false);
      setShowCustomGoogleInput(false);
      setCustomGoogleEmail('');
    }
  }, [isAuthModalOpen, authModalMode]);

  if (!isAuthModalOpen) return null;

  // Real-time password requirement heuristics (NO uppercase per user instruction)
  const hasLength = password.length >= 8;
  const hasNumber = /[0-9]/.test(password);
  const hasSpecial = /[!@#$%^&*(),.?":{}|<>_\-+=\\/~`[\]]/.test(password);

  const rulesPassed = (hasLength ? 1 : 0) + (hasNumber ? 1 : 0) + (hasSpecial ? 1 : 0);

  let strengthPercent = 0;
  let strengthLabel = 'WEAK';
  let strengthColor = '#64748b';

  if (password.length > 0) {
    if (rulesPassed === 1) {
      strengthPercent = 35;
      strengthLabel = 'WEAK';
      strengthColor = '#ef4444';
    } else if (rulesPassed === 2) {
      strengthPercent = 70;
      strengthLabel = 'FAIR';
      strengthColor = '#f59e0b';
    } else if (rulesPassed === 3) {
      // Show 80% if length is exactly 8-9 chars (matching reference image), 100% if longer
      strengthPercent = password.length >= 10 ? 100 : 80;
      strengthLabel = 'STRONG';
      strengthColor = '#10b981';
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail || !cleanEmail.includes('@')) {
      setError('Please provide a valid email address.');
      return;
    }

    if (isSignUp) {
      if (!hasLength) {
        setError('Password must be at least 8 characters long.');
        return;
      }
      if (!hasNumber) {
        setError('Password must contain at least 1 number.');
        return;
      }
      if (!hasSpecial) {
        setError('Password must contain at least 1 special character.');
        return;
      }
      if (password !== confirm) {
        setError('Passwords do not match.');
        return;
      }

      // Format a pleasant full name from email prefix
      const derivedName = cleanEmail
        .split('@')[0]
        .replace(/[._+-]/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase()) || 'Healthcare Worker';

      setLoading(true);
      try {
        await register({
          name: derivedName,
          email: cleanEmail,
          password,
          confirm_password: confirm,
          role: 'health_worker'
        });
        closeAuthModal();
      } catch (err) {
        setError(err.message || 'Registration failed. Please check your details.');
      } finally {
        setLoading(false);
      }
    } else {
      if (!password) {
        setError('Please enter your password.');
        return;
      }

      setLoading(true);
      try {
        await login(cleanEmail, password);
        closeAuthModal();
      } catch (err) {
        setError(err.message || 'Invalid email or password.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleGoogleAccountSelect = async (account) => {
    setError('');
    setLoading(true);
    try {
      await googleLogin({
        email: account.email,
        name: account.name
      });
      closeAuthModal();
    } catch (err) {
      setError(err.message || 'Google sign-in failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCustomGoogleSubmit = async (e) => {
    e.preventDefault();
    const clean = customGoogleEmail.trim().toLowerCase();
    if (!clean || !clean.includes('@')) {
      setError('Please enter a valid Google email address.');
      return;
    }
    const namePart = clean
      .split('@')[0]
      .replace(/[._+-]/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
    await handleGoogleAccountSelect({ email: clean, name: namePart });
  };

  return (
    <div
      className="auth-modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) closeAuthModal();
      }}
    >
      <section
        className="auth-modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
      >
        {/* Close Button */}
        <button
          type="button"
          className="auth-modal-close"
          aria-label="Close"
          onClick={closeAuthModal}
        >
          <X size={18} />
        </button>

        {showGoogleChooser ? (
          /* Google Account Chooser View */
          <div className="google-chooser-container">
            <div className="google-chooser-logo">
              <svg width="34" height="34" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="#4285F4"
                  d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z"
                />
                <path
                  fill="#34A853"
                  d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.36 24 12 24z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.16 0 9.98 0 12s.45 3.84 1.25 5.42l4.03-3.15z"
                />
                <path
                  fill="#EA4335"
                  d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.36 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"
                />
              </svg>
            </div>
            <h2 className="google-chooser-title">Choose an account</h2>
            <p className="google-chooser-sub">to continue to NutriSense AI</p>

            {error && (
              <div role="alert" className="auth-modal-alert error">
                <AlertCircle size={15} />
                <span>{error}</span>
              </div>
            )}

            {!showCustomGoogleInput ? (
              <div className="google-accounts-list">
                <button
                  type="button"
                  className="google-account-item"
                  onClick={() =>
                    handleGoogleAccountSelect({
                      name: 'Anirudh',
                      email: 'anirudh.researcher@gmail.com'
                    })
                  }
                  disabled={loading}
                >
                  <div
                    className="google-account-avatar"
                    style={{ background: 'linear-gradient(135deg, #2563eb, #38bdf8)' }}
                  >
                    A
                  </div>
                  <div className="google-account-info">
                    <span className="google-account-name">Anirudh</span>
                    <span className="google-account-email">anirudh.researcher@gmail.com</span>
                  </div>
                  <ChevronRight size={16} className="google-chevron" />
                </button>

                <button
                  type="button"
                  className="google-account-item"
                  onClick={() =>
                    handleGoogleAccountSelect({
                      name: 'Sunita Devi (ASHA)',
                      email: 'asha.sunita@health.gov.in'
                    })
                  }
                  disabled={loading}
                >
                  <div
                    className="google-account-avatar"
                    style={{ background: 'linear-gradient(135deg, #059669, #10b981)' }}
                  >
                    S
                  </div>
                  <div className="google-account-info">
                    <span className="google-account-name">Sunita Devi (ASHA)</span>
                    <span className="google-account-email">asha.sunita@health.gov.in</span>
                  </div>
                  <ChevronRight size={16} className="google-chevron" />
                </button>

                <button
                  type="button"
                  className="google-account-item"
                  onClick={() => setShowCustomGoogleInput(true)}
                  disabled={loading}
                >
                  <div className="google-account-avatar another">
                    <User size={18} />
                  </div>
                  <div className="google-account-info">
                    <span className="google-account-name">Use another account</span>
                    <span className="google-account-email">Sign in with custom Google ID</span>
                  </div>
                  <ChevronRight size={16} className="google-chevron" />
                </button>
              </div>
            ) : (
              <form onSubmit={handleCustomGoogleSubmit} className="google-custom-form">
                <div className="auth-modal-field" style={{ marginBottom: '1rem' }}>
                  <label htmlFor="custom-google-email">GOOGLE EMAIL ADDRESS</label>
                  <input
                    id="custom-google-email"
                    type="email"
                    placeholder="yourname@gmail.com"
                    value={customGoogleEmail}
                    onChange={(e) => setCustomGoogleEmail(e.target.value)}
                    required
                    autoFocus
                  />
                </div>
                <button
                  type="submit"
                  className="auth-modal-submit-btn"
                  disabled={loading}
                >
                  {loading ? 'Signing in…' : 'Continue with this account'}
                </button>
              </form>
            )}

            <button
              type="button"
              className="google-chooser-back"
              onClick={() => {
                setShowGoogleChooser(false);
                setShowCustomGoogleInput(false);
                setError('');
              }}
            >
              <ArrowLeft size={14} />
              <span>Back to standard login</span>
            </button>
          </div>
        ) : (
          /* Standard Auth Form (Sign In / Sign Up) */
          <>
            {/* Top Pill Switcher */}
            <div className="auth-pill-switcher" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={!isSignUp}
                className={`auth-pill-btn ${!isSignUp ? 'active' : ''}`}
                onClick={() => {
                  setError('');
                  setAuthModalMode('signin');
                }}
              >
                SIGN IN
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={isSignUp}
                className={`auth-pill-btn ${isSignUp ? 'active' : ''}`}
                onClick={() => {
                  setError('');
                  setAuthModalMode('signup');
                }}
              >
                SIGN UP
              </button>
            </div>

            {/* Header Titles */}
            <div className="auth-modal-heading-wrap">
              <h2 id="auth-modal-title" className="auth-modal-main-title">
                {isSignUp ? 'Create account' : 'Welcome back'}
              </h2>
              <p className="auth-modal-sub-link">
                {isSignUp ? 'join nutrisense today →' : 'sign in to continue →'}
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div role="alert" className="auth-modal-alert error">
                <AlertCircle size={15} />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="auth-modal-form" noValidate>
              {/* Email Address */}
              <div className="auth-modal-field">
                <label htmlFor="auth-modal-email">EMAIL</label>
                <input
                  id="auth-modal-email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                />
              </div>

              {isSignUp ? (
                /* Sign Up: Side-by-side Password and Confirm */
                <>
                  <div className="auth-modal-pw-grid">
                    <div className="auth-modal-field">
                      <label htmlFor="auth-modal-password">PASSWORD</label>
                      <div className="auth-modal-pw-wrap">
                        <input
                          id="auth-modal-password"
                          type={showPassword ? 'text' : 'password'}
                          autoComplete="new-password"
                          required
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          disabled={loading}
                        />
                        <button
                          type="button"
                          className="auth-modal-eye-btn"
                          aria-label={showPassword ? 'Hide password' : 'Show password'}
                          onClick={() => setShowPassword((v) => !v)}
                          tabIndex={-1}
                        >
                          {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>

                    <div className="auth-modal-field">
                      <label htmlFor="auth-modal-confirm">CONFIRM</label>
                      <div className="auth-modal-pw-wrap">
                        <input
                          id="auth-modal-confirm"
                          type={showConfirm ? 'text' : 'password'}
                          autoComplete="new-password"
                          required
                          value={confirm}
                          onChange={(e) => setConfirm(e.target.value)}
                          disabled={loading}
                        />
                        <button
                          type="button"
                          className="auth-modal-eye-btn"
                          aria-label={showConfirm ? 'Hide password' : 'Show password'}
                          onClick={() => setShowConfirm((v) => !v)}
                          tabIndex={-1}
                        >
                          {showConfirm ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Password Strength Meter Card (No 1 Uppercase Requirement) */}
                  <div className="auth-strength-card">
                    <div className="auth-strength-top-row">
                      <span
                        className="auth-strength-title"
                        style={{ color: password.length > 0 ? strengthColor : '#9ca3af' }}
                      >
                        {strengthLabel}
                      </span>
                      <span
                        className="auth-strength-pct-label"
                        style={{ color: password.length > 0 ? strengthColor : '#9ca3af' }}
                      >
                        {strengthPercent}%
                      </span>
                    </div>

                    <div className="auth-strength-track">
                      <div
                        className="auth-strength-fill"
                        style={{
                          width: `${strengthPercent}%`,
                          backgroundColor: strengthColor
                        }}
                      />
                    </div>

                    <div className="auth-strength-checklist">
                      <div className={`auth-checklist-item ${hasLength ? 'met' : ''}`}>
                        <span className="auth-checklist-badge">
                          {hasLength ? (
                            <Check size={11} strokeWidth={3.5} />
                          ) : (
                            <span className="auth-badge-dot-inner" />
                          )}
                        </span>
                        <span>8+ characters</span>
                      </div>

                      <div className={`auth-checklist-item ${hasSpecial ? 'met' : ''}`}>
                        <span className="auth-checklist-badge">
                          {hasSpecial ? (
                            <Check size={11} strokeWidth={3.5} />
                          ) : (
                            <span className="auth-badge-dot-inner" />
                          )}
                        </span>
                        <span>1 special char</span>
                      </div>

                      <div className={`auth-checklist-item ${hasNumber ? 'met' : ''}`}>
                        <span className="auth-checklist-badge">
                          {hasNumber ? (
                            <Check size={11} strokeWidth={3.5} />
                          ) : (
                            <span className="auth-badge-dot-inner" />
                          )}
                        </span>
                        <span>1 number</span>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                /* Sign In: Single Password field with Forgot link */
                <div className="auth-modal-field">
                  <label htmlFor="auth-modal-password">PASSWORD</label>
                  <div className="auth-modal-pw-wrap">
                    <input
                      id="auth-modal-password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={loading}
                    />
                    <button
                      type="button"
                      className="auth-modal-eye-btn"
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                      onClick={() => setShowPassword((v) => !v)}
                      tabIndex={-1}
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              )}

              {/* Solid White CTA Button */}
              <button
                type="submit"
                className="auth-modal-submit-btn"
                disabled={loading}
              >
                {loading
                  ? 'Please wait…'
                  : isSignUp
                  ? 'CREATE ACCOUNT'
                  : 'SIGN IN'}
              </button>
            </form>

            {/* Divider */}
            <div className="auth-modal-divider">
              <span>OR CONTINUE WITH</span>
            </div>

            {/* Google SSO Button */}
            <button
              type="button"
              className="auth-modal-google-btn"
              onClick={() => {
                setError('');
                setShowGoogleChooser(true);
              }}
              disabled={loading}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="#4285F4"
                  d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.66-5.17 3.66-9.17z"
                />
                <path
                  fill="#34A853"
                  d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.1-6.72-4.93H1.25v3.15C3.26 21.36 7.36 24 12 24z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.28 14.27c-.25-.72-.38-1.49-.38-2.27s.13-1.55.38-2.27V6.58H1.25C.45 8.16 0 9.98 0 12s.45 3.84 1.25 5.42l4.03-3.15z"
                />
                <path
                  fill="#EA4335"
                  d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.36 0 3.26 2.64 1.25 6.58l4.03 3.15c.95-2.83 3.6-4.98 6.72-4.98z"
                />
              </svg>
              <span>Continue with Google</span>
            </button>

            {/* Terms and Privacy Policy (Footnote Removed!) */}
            <p className="auth-modal-legal">
              {isSignUp ? (
                <>
                  By creating an account, you agree to our{' '}
                  <a href="#terms" onClick={(e) => e.preventDefault()}>
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a href="#privacy" onClick={(e) => e.preventDefault()}>
                    Privacy Policy
                  </a>
                  .
                </>
              ) : (
                <>
                  By signing in, you agree to our{' '}
                  <a href="#terms" onClick={(e) => e.preventDefault()}>
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a href="#privacy" onClick={(e) => e.preventDefault()}>
                    Privacy Policy
                  </a>
                  .
                </>
              )}
            </p>
          </>
        )}
      </section>
    </div>
  );
}
