import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, ShieldAlert, HeartPulse, BarChart3 } from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function HomePage() {
  return (
    <div className="container" style={{ paddingTop: '2rem' }}>
      {/* Top Disclaimer Banner */}
      <DisclaimerBanner />

      {/* Hero Section */}
      <section style={{ textAlign: 'center', maxWidth: '850px', margin: '1.5rem auto 3rem' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.85rem',
          backgroundColor: 'var(--color-primary-light)',
          color: 'var(--color-primary)',
          borderRadius: 'var(--radius-full)',
          fontSize: '0.85rem',
          fontWeight: 700,
          marginBottom: '1.25rem',
          maxWidth: '100%'
        }}>
          <Sparkles size={16} style={{ flexShrink: 0 }} />
          <span>Scenario A — Scale-Free Community Pre-Screening</span>
        </div>

        <h1 className="hero-title" style={{ marginBottom: '1.25rem' }}>
          Childhood Malnutrition Risk Intelligence & Early Triage
        </h1>

        <p className="hero-lead" style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
          NutriSense AI evaluates early childhood undernutrition risk without requiring weighing scales
          or stadiometers. Using non-invasive demographic, maternal, household, and recent morbidity
          markers, our calibrated machine learning models provide rapid risk intelligence for community health workers.
        </p>

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/screen" className="btn btn-primary btn-lg">
            <span>Start Child Screening</span>
            <ArrowRight size={18} />
          </Link>
          <Link to="/about" className="btn btn-secondary btn-lg">
            <span>Methodology & Research</span>
          </Link>
        </div>
      </section>

      {/* Tri-Target Condition Cards */}
      <section style={{ marginBottom: '4rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h2 style={{ marginBottom: '0.5rem' }}>Tri-Target Undernutrition Assessment</h2>
          <p style={{ color: 'var(--text-muted)' }}>
            Evaluates statistical risk across all three internationally recognized pediatric undernutrition conditions.
          </p>
        </div>

        <div className="results-grid">
          {/* Stunting Card */}
          <div className="card" style={{ borderTop: '4px solid var(--color-primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--color-primary-light)',
                color: 'var(--color-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <HeartPulse size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>Stunting Risk</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Chronic Undernutrition (HAZ &lt; -2 SD)</span>
              </div>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Reflects cumulative, long-term linear growth deficits caused by chronic recurrent illness,
              inadequate dietary diversity, and adverse household sanitation.
            </p>
            <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.825rem', color: 'var(--color-primary)', fontWeight: 600 }}>
              Operating Decision Threshold: 35.0%
            </div>
          </div>

          {/* Underweight Card */}
          <div className="card" style={{ borderTop: '4px solid #0ea5e9' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#f0f9ff',
                color: '#0ea5e9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <BarChart3 size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>Underweight Risk</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Composite Deficit (WAZ &lt; -2 SD)</span>
              </div>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              A composite measure capturing both chronic linear growth failure and acute body mass
              depletion. Sensitive to maternal nutrition and infant feeding practices.
            </p>
            <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.825rem', color: '#0ea5e9', fontWeight: 600 }}>
              Operating Decision Threshold: 31.0%
            </div>
          </div>

          {/* Wasting Card */}
          <div className="card" style={{ borderTop: '4px solid #f59e0b' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: '#fffbeb',
                color: '#f59e0b',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <ShieldAlert size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>Wasting Risk</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Acute Undernutrition (WHZ &lt; -2 SD)</span>
              </div>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Signals rapid, acute weight loss or failure to gain weight, frequently triggered by recent
              diarrhea, high fever, or severe food insecurity.
            </p>
            <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.825rem', color: '#f59e0b', fontWeight: 600 }}>
              Operating Decision Threshold: 17.0%
            </div>
          </div>
        </div>
      </section>

      {/* How it Works / Workflow */}
      <section style={{ marginBottom: '4rem' }}>
        <div className="card" style={{ padding: 'clamp(1.25rem, 3vw, 2.5rem)' }}>
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h2 style={{ marginBottom: '0.5rem' }}>How Community Pre-Screening Operates</h2>
            <p style={{ color: 'var(--text-muted)' }}>
              Non-invasive, scale-free workflow designed for frontline health workers in low-resource settings.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 240px), 1fr))', gap: '1.5rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <span style={{
                  width: '2rem', height: '2rem', borderRadius: 'var(--radius-full)',
                  backgroundColor: 'var(--color-primary-light)', color: 'var(--color-primary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800
                }}>1</span>
                <h4 style={{ fontSize: '1.1rem' }}>Collect Non-Invasive Data</h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                Health workers collect standard demographic, maternal history, household environment,
                and 2-week symptom recall without physical instruments.
              </p>
            </div>

            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <span style={{
                  width: '2rem', height: '2rem', borderRadius: 'var(--radius-full)',
                  backgroundColor: 'var(--color-primary-light)', color: 'var(--color-primary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800
                }}>2</span>
                <h4 style={{ fontSize: '1.1rem' }}>LightGBM Model Inference</h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                Approved LightGBM models trained on NFHS-5 calculate unrounded risk probabilities for
                stunting, underweight, and wasting in milliseconds.
              </p>
            </div>

            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <span style={{
                  width: '2rem', height: '2rem', borderRadius: 'var(--radius-full)',
                  backgroundColor: 'var(--color-primary-light)', color: 'var(--color-primary)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800
                }}>3</span>
                <h4 style={{ fontSize: '1.1rem' }}>Calibrated Decision Triage</h4>
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                Pre-specified sensitivity-optimized decision thresholds flag children who are
                screen-positive for prompt referral to formal anthropometric evaluation.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
