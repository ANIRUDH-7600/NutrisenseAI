import React from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  ShieldAlert,
  HeartPulse,
  BarChart3
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function HomePage() {
  return (
    <div className="container" style={{ paddingTop: '2rem' }}>
      {/* Top Disclaimer Banner */}
      <DisclaimerBanner />

      {/* Split 2-Column Hero Section (Matching SmartDocQ Design) */}
      <section className="hero-split-grid">
        {/* Left Column: Bold Typography & Supported Targets */}
        <div className="hero-left-col">
          <h1 className="smart-hero-title">
            Your community.<br />
            Now you can <em>screen</em> them.
          </h1>

          <div className="supported-meta-row">
            <span className="supported-meta-label">SUPPORTED SCREENING TARGETS</span>
            <span className="supported-line" />
          </div>

          <div className="supported-chips">
            <span>STUNTING</span>
            <span>UNDERWEIGHT</span>
            <span>WASTING</span>
            <span>HAZ</span>
            <span>WAZ</span>
            <span>WHZ</span>
          </div>

          <div className="supported-desc-group">
            <p className="supported-desc-line">Screen childhood undernutrition across your community without physical scales.</p>
            <p className="supported-desc-line">Trace risk back to calibrated physiological and household etiology.</p>
          </div>

          <Link to="/screen" className="btn-open-smart">
            <span>OPEN NUTRISENSE</span>
            <ArrowRight size={15} />
          </Link>
        </div>

        {/* Right Column: Realistic Paper Dossier Document + Side Explore Cards */}
        <div className="paper-dossier-wrap">
          {/* Authentic Paper Document Card */}
          <div className="paper-dossier">
            <div className="paper-dossier-tag">
              CASE 04 // PEDIATRIC_TRIAGE.MED
            </div>

            <div className="paper-label">QUESTION</div>
            <div className="paper-question">
              What is non-invasive stunting risk?
            </div>

            <div className="paper-label">ANSWER</div>
            <p className="paper-answer">
              Stunting is the process by which chronic recurrent illness and adverse household environment restrict <mark className="paper-highlight">linear growth</mark> (HAZ &lt; -2 SD).
              Non-invasive ML markers detect early biological faltering before physical stadiometers or calibrated scales are accessible.
            </p>

            {/* Technical Diagram */}
            <div className="paper-diagram">
              <svg width="100%" height="45" viewBox="0 0 240 45" fill="none" xmlns="http://www.w3.org/2000/svg">
                {/* Horizontal baseline */}
                <line x1="10" y1="35" x2="230" y2="35" stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 3" />
                {/* Threshold Marker */}
                <line x1="140" y1="5" x2="140" y2="38" stroke="#0f172a" strokeWidth="1.5" />
                {/* Bell-like risk curve */}
                <path d="M 15 35 Q 80 34 110 20 Q 140 5 160 12 Q 190 28 225 35" stroke="#0f172a" strokeWidth="1.5" fill="none" />
                <circle cx="140" cy="8" r="3" fill="#0f172a" />
              </svg>
              <div className="paper-diagram-caption">
                Fig. 4.2 LightGBM calibrated decision boundary (&tau; = 0.35)
              </div>
            </div>

            <div className="paper-footer-meta">
              <span>Grounded in • India NFHS-5</span>
              <span>Scenario A Triage</span>
            </div>
          </div>

          {/* Floating Dark Side Explore Cards (Side-by-side with paper) */}
          <div className="side-explore-wrap">
            <div className="side-explore-label">
              <span className="sparkle-icon">✦</span>
              <span>EXPLORE ASSESSMENTS</span>
            </div>

            <Link to="/screen" className="side-explore-card">
              <div className="side-explore-header">
                <span className="explore-num">01</span>
                <span className="explore-action">ASK &rarr;</span>
              </div>
              <div className="side-explore-text">
                What role does maternal BMI play?
              </div>
            </Link>

            <Link to="/screen" className="side-explore-card">
              <div className="side-explore-header">
                <span className="explore-num">02</span>
                <span className="explore-action">ASK &rarr;</span>
              </div>
              <div className="side-explore-text">
                What triage threshold triggers stunting referral?
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Capabilities Section */}
      <section style={{ marginBottom: '6rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
          <div className="supported-meta-label">
            01 // CAPABILITIES
          </div>
          <h2 style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', color: '#ffffff', letterSpacing: '-0.02em', marginTop: '0.5rem' }}>
            Make Every Screening <span style={{ fontFamily: 'var(--font-serif)', fontStyle: 'italic', fontWeight: 400 }}>More Useful.</span>
          </h2>
          <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0.5rem auto 0', fontSize: '0.975rem' }}>
            Tri-target epidemiological risk evaluation across all three internationally recognized pediatric undernutrition conditions.
          </p>
        </div>

        {/* 3 Dark Slate Capability Cards */}
        <div className="results-grid">
          {/* Stunting Card */}
          <div className="card">
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: 'rgba(255, 255, 255, 0.6)',
              marginBottom: '1rem'
            }}>
              <span>01 // STUNTING</span>
              <span style={{ color: '#38bdf8' }}>HAZ &lt; -2 SD</span>
            </div>

            <div style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: '8px',
              height: '110px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1.25rem'
            }}>
              <div style={{
                width: '3rem',
                height: '3rem',
                borderRadius: '8px',
                background: 'rgba(56, 189, 248, 0.1)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#38bdf8'
              }}>
                <HeartPulse size={24} />
              </div>
            </div>

            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Chronic Linear Deficit.
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.25rem' }}>
              Captures cumulative linear growth failure caused by recurrent illness, dietary gaps, and environmental enteric dysfunction.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '0.85rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Referral Cutoff</span>
              <span style={{ color: '#ffffff', fontWeight: 700 }}>35.0%</span>
            </div>
          </div>

          {/* Underweight Card */}
          <div className="card">
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: 'rgba(255, 255, 255, 0.6)',
              marginBottom: '1rem'
            }}>
              <span>02 // UNDERWEIGHT</span>
              <span style={{ color: '#60a5fa' }}>WAZ &lt; -2 SD</span>
            </div>

            <div style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: '8px',
              height: '110px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1.25rem'
            }}>
              <div style={{
                width: '3rem',
                height: '3rem',
                borderRadius: '8px',
                background: 'rgba(96, 165, 250, 0.1)',
                border: '1px solid rgba(96, 165, 250, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#60a5fa'
              }}>
                <BarChart3 size={24} />
              </div>
            </div>

            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Composite Growth Depletion.
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.25rem' }}>
              Captures combined linear growth deficits and body mass depletion. Highly sensitive to maternal nutrition and infant feeding.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '0.85rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Referral Cutoff</span>
              <span style={{ color: '#ffffff', fontWeight: 700 }}>31.0%</span>
            </div>
          </div>

          {/* Wasting Card */}
          <div className="card">
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: 'rgba(255, 255, 255, 0.6)',
              marginBottom: '1rem'
            }}>
              <span>03 // WASTING</span>
              <span style={{ color: '#f59e0b' }}>WHZ &lt; -2 SD</span>
            </div>

            <div style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: '8px',
              height: '110px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '1.25rem'
            }}>
              <div style={{
                width: '3rem',
                height: '3rem',
                borderRadius: '8px',
                background: 'rgba(245, 158, 11, 0.1)',
                border: '1px solid rgba(245, 158, 11, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#f59e0b'
              }}>
                <ShieldAlert size={24} />
              </div>
            </div>

            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Acute Tissue Loss & Triage.
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.25rem' }}>
              Identifies rapid weight loss and severe acute vulnerability, frequently triggered by recent diarrhea, fever, or food shock.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '0.85rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Referral Cutoff</span>
              <span style={{ color: '#ffffff', fontWeight: 700 }}>17.0%</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
