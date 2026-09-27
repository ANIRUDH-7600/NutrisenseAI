import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Menu, X, ArrowUpRight } from 'lucide-react';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <nav className="navbar" aria-label="Main Navigation">
      <div className="container navbar-inner">
        {/* Left: Brand */}
        <Link to="/" className="navbar-brand" onClick={closeMenu}>
          <img src="/logo.png" alt="NutriSense AI" className="navbar-logo-img" />
          <span className="brand-title">NutriSense AI</span>
        </Link>

        {/* Center: Clean Minimal Links with SmartDocQ Underline */}
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
        </ul>

        {/* Right: Sleek Scenario-A Pill Badge + User / AI Avatar */}
        <div className="navbar-right-action">
          <Link to="/screen" className="navbar-cta-pill" onClick={closeMenu}>
            <span>Scenario A</span>
            <ArrowUpRight size={13} />
          </Link>

          {/* SmartDocQ style circular avatar/badge */}
          <div className="navbar-avatar-circle" title="System Ready">
            <span className="avatar-initials">AI</span>
            <span className="avatar-status-dot" />
          </div>

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

