import React, { useState, useRef, useEffect } from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { Menu, X, LogOut, User, UserPlus, ChevronDown, ArrowRight, ClipboardList } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const dropdownRef = useRef(null);
  const navigate = useNavigate();

  const { isAuthenticated, user, logout, openAuthModal } = useAuth();

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => {
    setIsOpen(false);
    setShowUserDropdown(false);
  };

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowUserDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    closeMenu();
    await logout();
    navigate('/');
  };

  // Helper for clean initials (e.g. "AN", "AB", "SU")
  const getInitials = (name) => {
    if (!name) return 'HW';
    const clean = name.replace(/@.*$/, '').trim();
    const parts = clean.split(/[\s._-]+/).filter(Boolean);
    if (parts.length >= 2 && parts[0] && parts[1]) {
      return (parts[0][0] + parts[1][0]).toUpperCase();
    }
    return clean.slice(0, 2).toUpperCase();
  };

  // Helper for friendly first name / username
  const getDisplayName = (name) => {
    if (!name) return 'Account';
    const clean = name.replace(/@.*$/, '').trim();
    const parts = clean.split(/[\s._-]+/).filter(Boolean);
    return parts[0] || 'Account';
  };

  return (
    <nav className="navbar" aria-label="Main Navigation">
      <div className="container navbar-inner">
        {/* Left: Brand */}
        <Link to="/" className="navbar-brand" onClick={closeMenu}>
          <img src="/logo.png" alt="NutriSense AI" className="navbar-logo-img" />
          <span className="brand-title">NutriSense AI</span>
        </Link>

        {/* Center: Navigation Links */}
        <ul className={`nav-links ${isOpen ? 'open' : ''}`}>
          <li>
            <NavLink
              to="/"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
              end
            >
              Home
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/screen"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
            >
              Screen Child
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/results"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
            >
              Results
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/about"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
            >
              Methodology
            </NavLink>
          </li>

          {/* Mobile-Only Auth Links */}
          <li className="mobile-auth-item">
            {isAuthenticated && user ? (
              <div className="mobile-user-box">
                <div className="mobile-user-info">
                  <span className="mobile-user-name">{user.name}</span>
                  <span className="mobile-user-email-text">{user.email}</span>
                </div>
                <button onClick={handleLogout} className="mobile-logout-btn">
                  <LogOut size={16} /> Sign Out
                </button>
              </div>
            ) : (
              <div className="mobile-auth-actions">
                <button
                  type="button"
                  className="btn btn-secondary mobile-auth-btn"
                  onClick={() => { closeMenu(); openAuthModal('signin'); }}
                >
                  <User size={15} /> Log In
                </button>
                <button
                  type="button"
                  className="btn btn-primary mobile-auth-btn"
                  onClick={() => { closeMenu(); openAuthModal('signup'); }}
                >
                  <UserPlus size={15} /> Sign Up
                </button>
              </div>
            )}
          </li>
        </ul>

        {/* Right Action Bar */}
        <div className="navbar-right-action">
          {/* Authenticated State: User Dropdown */}
          {isAuthenticated && user ? (
            <div className="navbar-user-menu-wrapper" ref={dropdownRef}>
              <button
                type="button"
                className="navbar-user-pill-btn"
                onClick={() => setShowUserDropdown(!showUserDropdown)}
                aria-expanded={showUserDropdown}
                aria-haspopup="true"
                id="user-menu-button"
              >
                <div className="navbar-avatar-circle" title={user.name}>
                  <span className="avatar-initials">{getInitials(user.name)}</span>
                  <span className="avatar-status-dot" />
                </div>
                <span className="navbar-user-name-text">{getDisplayName(user.name)}</span>
                <ChevronDown size={14} className={`dropdown-chevron ${showUserDropdown ? 'rotate' : ''}`} />
              </button>

              {showUserDropdown && (
                <div className="navbar-user-dropdown" role="menu">
                  <div className="dropdown-user-header">
                    <p className="dropdown-user-name">{user.name}</p>
                    <p className="dropdown-user-email">{user.email}</p>
                  </div>
                  <div className="dropdown-divider" />
                  <Link
                    to="/screen"
                    className="dropdown-menu-item"
                    role="menuitem"
                    onClick={() => setShowUserDropdown(false)}
                  >
                    <User size={15} />
                    <span>Screen Child</span>
                  </Link>
                  <Link
                    to="/results"
                    className="dropdown-menu-item"
                    role="menuitem"
                    onClick={() => setShowUserDropdown(false)}
                  >
                    <ClipboardList size={15} />
                    <span>Screening Records</span>
                  </Link>
                  <div className="dropdown-divider" />
                  <button
                    type="button"
                    className="dropdown-menu-item logout-item"
                    role="menuitem"
                    onClick={handleLogout}
                  >
                    <LogOut size={15} />
                    <span>Sign Out</span>
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* Unauthenticated State: Minimal Luxury Pill Button (Image 4) */
            <button
              type="button"
              onClick={() => openAuthModal('signin')}
              className="navbar-login-pill-btn"
              id="nav-login-btn"
            >
              <User size={15} />
              <span>Log In</span>
              <ArrowRight size={14} />
            </button>
          )}

          <button
            className="mobile-menu-btn"
            onClick={toggleMenu}
            aria-expanded={isOpen}
            aria-label="Toggle navigation menu"
          >
            {isOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>
    </nav>
  );
}
