import React from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  ArrowRight,
  ShieldAlert,
  HeartPulse,
  BarChart3,
  Activity,
  Layers,
  CheckCircle2,
  FileCheck,
  Stethoscope
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function HomePage() {
  return (
    <div className="container" style={{ paddingTop: '2.5rem' }}>
      {/* Top Disclaimer Banner */}
      <DisclaimerBanner />

      {/* Hero Section */}
      <section style={{ textAlign: 'center', maxWidth: '880px', margin: '2.5rem auto 4.5rem', position: 'relative' }}>
        {/* Subtle Decorative Star Sparkles */}
        <div style={{ position: 'absolute', top: '-10px', left: '10%', opacity: 0.6 }} className="sparkle-accent">
          ✦
        </div>
        <div style={{ position: 'absolute', top: '40px', right: '8%', opacity: 0.7 }} className="sparkle-accent">
          ✦
        </div>

        {/* Terminal Micro-Tag */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div className="tag-terminal">
            <Sparkles size={13} />
            <span>✦ AI HEALTH INTELLIGENCE // SCENARIO A</span>
          </div>
        </div>

        {/* Hero Title */}
        <h1 className="hero-title" style={{ marginBottom: '1.5rem' }}>
          Predict Childhood Risk.<br />
          Now Screened <span className="text-serif-italic">More Accurately.</span>
        </h1>

        <p className="hero-lead" style={{ maxWidth: '720px', margin: '0 auto 2.5rem' }}>
          Evaluate early childhood undernutrition risk without requiring weighing scales or stadiometers.
          Non-invasive maternal, household, and symptom markers powered by sensitivity-calibrated
          LightGBM models for frontline community health workers.
        </p>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/screen" className="btn btn-pill-white btn-lg">
            <span>Start Child Screening</span>
            <ArrowRight size={18} />
          </Link>
          <Link to="/about" className="btn btn-pill-dark btn-lg">
            <span>Methodology & Research</span>
          </Link>
        </div>
      </section>

      {/* Capabilities Section Header (Matching Reference Design) */}
      <section style={{ marginBottom: '5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div style={{ marginBottom: '1rem' }}>
            <div className="tag-terminal">
              <span>01 // CAPABILITIES</span>
            </div>
          </div>
          <h2 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '0.75rem', color: '#ffffff' }}>
            Make Every Screening <span className="text-serif-italic">More Useful.</span>
          </h2>
          <p style={{ color: 'var(--text-muted)', maxWidth: '620px', margin: '0 auto', fontSize: '0.975rem' }}>
            Tri-target epidemiological risk assessment across the three internationally recognized pediatric undernutrition conditions.
          </p>
        </div>

        {/* 3 Capabilities Dark Showcase Cards */}
        <div className="results-grid">
          {/* Card 1: Stunting */}
          <div className="card">
            {/* Top Monospace Tag */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: '#38bdf8',
              letterSpacing: '0.06em',
              marginBottom: '1rem'
            }}>
              <span>01 // STUNTING</span>
              <span style={{ opacity: 0.6 }}>HAZ &lt; -2 SD</span>
            </div>

            {/* Inner Grid Showcase Graphic Box */}
            <div className="card-grid-preview">
              <div style={{
                width: '3.25rem',
                height: '3.25rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(56, 189, 248, 0.12)',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#38bdf8',
                boxShadow: '0 0 20px rgba(56, 189, 248, 0.25)'
              }}>
                <HeartPulse size={26} />
              </div>
            </div>

            {/* Title & Body */}
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Chronic Growth. <span style={{ color: '#38bdf8' }}>Linear Deficit.</span>
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.5rem' }}>
              Detects cumulative linear growth failure caused by recurrent illness, poor dietary diversity, and household sanitation barriers.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '1rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.775rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Decision Threshold</span>
              <span style={{ color: '#38bdf8', fontWeight: 700 }}>35.0%</span>
            </div>
          </div>

          {/* Card 2: Underweight */}
          <div className="card">
            {/* Top Monospace Tag */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: '#60a5fa',
              letterSpacing: '0.06em',
              marginBottom: '1rem'
            }}>
              <span>02 // UNDERWEIGHT</span>
              <span style={{ opacity: 0.6 }}>WAZ &lt; -2 SD</span>
            </div>

            {/* Inner Grid Showcase Graphic Box */}
            <div className="card-grid-preview">
              <div style={{
                width: '3.25rem',
                height: '3.25rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(96, 165, 250, 0.12)',
                border: '1px solid rgba(96, 165, 250, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#60a5fa',
                boxShadow: '0 0 20px rgba(96, 165, 250, 0.25)'
              }}>
                <BarChart3 size={26} />
              </div>
            </div>

            {/* Title & Body */}
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Composite Deficit. <span style={{ color: '#60a5fa' }}>Dual Factor.</span>
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.5rem' }}>
              Captures acute mass depletion and chronic growth faltering. Highly sensitive to maternal nutrition and infant feeding patterns.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '1rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.775rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Decision Threshold</span>
              <span style={{ color: '#60a5fa', fontWeight: 700 }}>31.0%</span>
            </div>
          </div>

          {/* Card 3: Wasting */}
          <div className="card">
            {/* Top Monospace Tag */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              fontWeight: 700,
              color: '#f59e0b',
              letterSpacing: '0.06em',
              marginBottom: '1rem'
            }}>
              <span>03 // WASTING</span>
              <span style={{ opacity: 0.6 }}>WHZ &lt; -2 SD</span>
            </div>

            {/* Inner Grid Showcase Graphic Box */}
            <div className="card-grid-preview">
              <div style={{
                width: '3.25rem',
                height: '3.25rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(245, 158, 11, 0.12)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#f59e0b',
                boxShadow: '0 0 20px rgba(245, 158, 11, 0.25)'
              }}>
                <ShieldAlert size={26} />
              </div>
            </div>

            {/* Title & Body */}
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: '#ffffff' }}>
              Acute Depletion. <span style={{ color: '#f59e0b' }}>Rapid Triage.</span>
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.55', marginBottom: '1.5rem' }}>
              Identifies acute tissue loss and high morbidity risk triggered by recent diarrhea, fever, or food insecurity.
            </p>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '1rem',
              borderTop: '1px solid var(--border-color)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.775rem'
            }}>
              <span style={{ color: 'var(--text-subtle)' }}>Decision Threshold</span>
              <span style={{ color: '#f59e0b', fontWeight: 700 }}>17.0%</span>
            </div>
          </div>
        </div>
      </section>

      {/* How Community Pre-Screening Operates */}
      <section style={{ marginBottom: '5rem' }}>
        <div className="card" style={{ padding: 'clamp(1.5rem, 4vw, 3rem)' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <div className="tag-terminal">
                <span>02 // OPERATIONAL ARCHITECTURE</span>
              </div>
            </div>
            <h2 style={{ fontSize: 'clamp(1.6rem, 3.5vw, 2.2rem)', marginBottom: '0.5rem', color: '#ffffff' }}>
              Community Pre-Screening. <span className="text-serif-italic">In Three Steps.</span>
            </h2>
            <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto', fontSize: '0.925rem' }}>
              Non-invasive, scale-free workflow engineered for frontline health workers in resource-constrained environments.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '2rem' }}>
            {/* Step 1 */}
            <div>
              <div style={{
                display: 'inline-block',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#38bdf8',
                marginBottom: '0.75rem',
                letterSpacing: '0.06em'
              }}>
                STEP 01 // INTERVIEW
              </div>
              <h4 style={{ fontSize: '1.15rem', color: '#ffffff', marginBottom: '0.5rem' }}>
                Collect Non-Invasive Data
              </h4>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                Frontline workers record standard maternal recall, child age, recent 2-week illnesses, and household attributes without requiring calibrated physical instruments.
              </p>
            </div>

            {/* Step 2 */}
            <div>
              <div style={{
                display: 'inline-block',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#818cf8',
                marginBottom: '0.75rem',
                letterSpacing: '0.06em'
              }}>
                STEP 02 // INFERENCE
              </div>
              <h4 style={{ fontSize: '1.15rem', color: '#ffffff', marginBottom: '0.5rem' }}>
                LightGBM Probability Engine
              </h4>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                Trained and cross-validated on India NFHS-5 data. Computes instantaneous, unrounded risk probabilities across stunting, underweight, and wasting in milliseconds.
              </p>
            </div>

            {/* Step 3 */}
            <div>
              <div style={{
                display: 'inline-block',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.75rem',
                fontWeight: 700,
                color: '#10b981',
                marginBottom: '0.75rem',
                letterSpacing: '0.06em'
              }}>
                STEP 03 // TRIAGE
              </div>
              <h4 style={{ fontSize: '1.15rem', color: '#ffffff', marginBottom: '0.5rem' }}>
                Calibrated Decision Boundary
              </h4>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                Sensitivity-calibrated decision thresholds classify children as screen-positive, prioritizing scarce infantometers and medical officer referrals for those at highest risk.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
