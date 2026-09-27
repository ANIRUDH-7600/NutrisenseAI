import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Activity, ShieldAlert, BookOpen, Menu, X, Sparkles } from 'lucide-react';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <nav className="navbar" aria-label="Main Navigation">
      <div className="container navbar-inner">
        <Link to="/" className="navbar-brand" onClick={closeMenu}>
          <div className="brand-icon-wrap">
            <Activity size={20} />
          </div>
          <span className="brand-title">NutriSense AI</span>
          <span className="brand-badge">Scenario A</span>
        </Link>

        <button
          className="mobile-menu-btn"
          onClick={toggleMenu}
          aria-expanded={isOpen}
          aria-label="Toggle navigation menu"
        >
          {isOpen ? <X size={22} /> : <Menu size={22} />}
        </button>

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
              <Sparkles size={16} />
              Screen Child
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/results"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
            >
              <ShieldAlert size={16} />
              Results
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/about"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              onClick={closeMenu}
            >
              <BookOpen size={16} />
              Methodology
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}
