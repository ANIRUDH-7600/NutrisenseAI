import React from 'react';
import { ShieldCheck, Heart } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer" role="contentinfo">
      <div className="container footer-content">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
          <ShieldCheck size={18} color="var(--color-primary)" />
          <span>NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence</span>
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
