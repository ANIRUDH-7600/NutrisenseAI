import React from 'react';
import { ShieldCheck, Heart } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer" role="contentinfo">
      <div className="container footer-content">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontWeight: 700, fontSize: '1.05rem', color: '#ffffff' }}>
          <img
            src="/logo.png"
            alt="NutriSense AI"
            style={{
              width: '2.5rem',
              height: '2.5rem',
              borderRadius: 'var(--radius-full)',
              border: '2px solid rgba(56, 189, 248, 0.4)',
              boxShadow: '0 0 12px rgba(56, 189, 248, 0.3)',
              background: '#ffffff'
            }}
          />
          <div>
            <span>NutriSense AI</span>
            <span style={{ display: 'block', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', letterSpacing: '0.08em', fontWeight: 600 }}>
              AI FOR HEALTHIER CHILDREN
            </span>
          </div>
        </div>
        <p style={{ maxWidth: '780px', color: 'var(--text-muted)', fontSize: '0.825rem' }}>
          Trained on India National Family Health Survey (NFHS-5, 2019–21). Configured exclusively for
          Scenario A community pre-screening triage. Direct anthropometric outcome variables are
          strictly isolated. No DHS microdata is bundled, accessed, or exposed at runtime.
        </p>
        <div style={{ fontSize: '0.775rem', color: 'var(--text-subtle)' }}>
          Research & Academic Final-Year Project • Built with rigorous epidemiological standards
        </div>
      </div>
    </footer>
  );
}
