import React from 'react';
import { useLocation, Link, useNavigate } from 'react-router-dom';
import {
  ShieldAlert,
  ShieldCheck,
  ArrowLeft,
  Printer,
  Info,
  AlertTriangle,
  HeartPulse,
  BarChart3,
  Stethoscope,
  Sparkles
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function ResultsPage({ screeningResult }) {
  const location = useLocation();
  const navigate = useNavigate();

  // Retrieve result from router state or props
  const resultData = location.state?.result || screeningResult;

  if (!resultData || !resultData.predictions) {
    return (
      <div className="container" style={{ paddingTop: '4rem', textAlign: 'center', maxWidth: '600px' }}>
        <div className="card" style={{ padding: '3rem 2rem' }}>
          <div style={{
            width: '3.5rem',
            height: '3.5rem',
            borderRadius: 'var(--radius-full)',
            backgroundColor: 'var(--color-primary-light)',
            color: 'var(--color-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.5rem'
          }}>
            <Sparkles size={28} />
          </div>
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>No Active Screening Results</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginBottom: '1.75rem' }}>
            To view childhood undernutrition risk intelligence, please complete the Scenario-A child screening form.
          </p>
          <Link to="/screen" className="btn btn-primary">
            Start Child Screening
          </Link>
        </div>
      </div>
    );
  }

  const { predictions, model_version, feature_schema_version } = resultData;
  const childName = resultData.child_name || inputPayload?.child_name || 'Anonymous Child';

  const handlePrint = () => {
    window.print();
  };

  const renderTargetCard = (targetKey, title, subtitle, iconComponent) => {
    const data = predictions[targetKey];
    if (!data) return null;

    const probPct = (data.probability * 100).toFixed(1);
    const threshPct = (data.threshold * 100).toFixed(1);
    const isPositive = data.screen_positive;

    return (
      <div key={targetKey} className={`result-card ${isPositive ? 'positive' : 'negative'}`}>
        <div>
          {/* Card Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <div style={{
                width: '2.5rem',
                height: '2.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: isPositive ? 'var(--color-danger-light)' : 'var(--color-success-light)',
                color: isPositive ? 'var(--color-danger)' : 'var(--color-success)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {iconComponent}
              </div>
              <div>
                <h3 style={{ fontSize: '1.25rem', textTransform: 'capitalize' }}>{title}</h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</span>
              </div>
            </div>

            <span className={`result-card-badge ${isPositive ? 'badge-positive' : 'badge-negative'}`}>
              {isPositive ? <ShieldAlert size={14} /> : <ShieldCheck size={14} />}
              {isPositive ? 'Screen Positive' : 'Screen Negative'}
            </span>
          </div>

          {/* Probability Metric */}
          <div className="metric-row">
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Predicted Probability
              </span>
              <div className="metric-value" style={{ color: isPositive ? 'var(--color-danger)' : 'var(--color-success)' }}>
                {probPct}%
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
                Unrounded: {data.probability.toFixed(6)}
              </span>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Screening Threshold
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-main)' }}>
                {threshPct}%
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
                Rule: Prob &gt;= {data.threshold}
              </span>
            </div>
          </div>

          {/* Visual Comparison Progress Bar */}
          <div className="progress-bar-wrap" title={`Probability: ${probPct}%, Threshold: ${threshPct}%`}>
            <div
              className={`progress-bar-fill ${isPositive ? 'fill-positive' : 'fill-negative'}`}
              style={{ width: `${Math.min(data.probability * 100, 100)}%` }}
            />
            {/* Threshold Line Marker */}
            <div
              className="threshold-marker"
              style={{ left: `${data.threshold * 100}%` }}
              title={`Decision Threshold: ${threshPct}%`}
            />
          </div>
        </div>

        {/* Interpretation & Model Info */}
        <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.45', marginBottom: '0.75rem' }}>
            {isPositive
              ? 'The predicted probability is above the community screening threshold, indicating elevated statistical risk. Secondary clinical anthropometric triage is indicated.'
              : 'The predicted probability is below the calibrated screening threshold. Continue standard community growth monitoring and infant feeding guidance.'}
          </p>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-subtle)' }}>
            <span>Model: {data.model_family} ({data.condition})</span>
            <span>Scenario A Triage</span>
          </div>
        </div>
      </div>
    );
  };

  const positiveCount = Object.values(predictions).filter((p) => p.screen_positive).length;

  return (
    <div className="container" style={{ paddingTop: '2rem', maxWidth: '1080px' }}>
      <DisclaimerBanner />

      {/* Results Header */}
      <div className="page-header-row">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem', flexWrap: 'wrap' }}>
            <h1>
              {childName && childName !== 'Anonymous Child' ? `Screening for ${childName}` : 'Screening Risk Dashboard'}
            </h1>
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 700,
              padding: '0.2rem 0.6rem',
              borderRadius: 'var(--radius-full)',
              backgroundColor: 'var(--bg-subtle)',
              color: 'var(--text-muted)'
            }}>
              {model_version}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.925rem' }}>
            {childName && childName !== 'Anonymous Child' && (
              <strong style={{ color: 'var(--color-primary)' }}>Child: {childName} • </strong>
            )}
            Non-invasive Scenario-A community risk evaluation across three pediatric undernutrition conditions.
          </p>
        </div>

        <div className="action-btn-group">
          <button onClick={handlePrint} className="btn btn-secondary btn-sm" title="Print results dashboard">
            <Printer size={15} />
            Print Summary
          </button>
          <Link to="/screen" className="btn btn-primary btn-sm">
            <ArrowLeft size={15} />
            New Screening
          </Link>
        </div>
      </div>

      {/* Summary Status Banner */}
      <div
        className="results-summary-banner"
        style={{
          backgroundColor: positiveCount > 0 ? 'var(--color-danger-light)' : 'var(--color-success-light)',
          border: `1px solid ${positiveCount > 0 ? 'var(--color-danger-border)' : 'var(--color-success-border)'}`
        }}
      >
        <div style={{
          width: '3rem',
          height: '3rem',
          borderRadius: 'var(--radius-full)',
          backgroundColor: '#fff',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: positiveCount > 0 ? 'var(--color-danger)' : 'var(--color-success)',
          flexShrink: 0
        }}>
          {positiveCount > 0 ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
        </div>
        <div>
          <strong style={{ fontSize: '1.1rem', color: positiveCount > 0 ? 'var(--color-danger)' : 'var(--color-success)' }}>
            {positiveCount > 0
              ? `${positiveCount} of 3 Undernutrition Indicators Flagged Screen-Positive`
              : 'All 3 Undernutrition Indicators Below Screening Threshold'}
          </strong>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-main)', marginTop: '0.2rem' }}>
            {positiveCount > 0
              ? 'This child presents elevated epidemiological risk. Referral for secondary triage and direct anthropometric measurements (Height, Weight, MUAC) is strongly recommended.'
              : 'Statistical risk is below operational thresholds across stunting, underweight, and wasting. Routine Anganwadi growth monitoring should be maintained.'}
          </p>
        </div>
      </div>

      {/* Results Cards Grid */}
      <div className="results-grid" style={{ marginBottom: '3rem' }}>
        {renderTargetCard('stunting', 'Stunting', 'Chronic Undernutrition (HAZ < -2 SD)', <HeartPulse size={22} />)}
        {renderTargetCard('underweight', 'Underweight', 'Composite Growth Deficit (WAZ < -2 SD)', <BarChart3 size={22} />)}
        {renderTargetCard('wasting', 'Wasting', 'Acute Malnutrition (WHZ < -2 SD)', <ShieldAlert size={22} />)}
      </div>

      {/* Secondary Triage Clinical Guidance Box */}
      <div className="card" style={{ marginBottom: '2.5rem', backgroundColor: '#f8fafc' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <Stethoscope size={20} color="var(--color-primary)" />
          <h3 style={{ fontSize: '1.15rem' }}>Secondary Triage & Follow-up Protocols</h3>
        </div>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: '1rem' }}>
          NutriSense AI is designed to prioritize limited community healthcare equipment by identifying children
          in need of immediate formal anthropometric assessment. If any indicator is flagged <strong>Screen Positive</strong>:
        </p>
        <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--text-main)', lineHeight: '1.6' }}>
          <li><strong>Direct Measurement:</strong> Measure recumbent length / standing height using an infantometer or stadiometer.</li>
          <li><strong>Direct Weighing:</strong> Record unclothed body weight using a calibrated digital hanging scale (SECA or UNISCALE standard).</li>
          <li><strong>Mid-Upper Arm Circumference (MUAC):</strong> Assess for acute wasting using standard three-color Shakir tape.</li>
          <li><strong>Bilateral Pitting Edema:</strong> Check both feet with thumb pressure for clinical signs of kwashiorkor.</li>
          <li><strong>Health Worker Referral:</strong> Escalate case to the Sector Medical Officer or Primary Health Centre (PHC) according to national POSHAN Abhiyaan guidelines.</li>
        </ul>
      </div>

      {/* Navigation Footer */}
      <div className="results-footer-nav">
        <Link to="/screen" className="btn btn-secondary">
          <ArrowLeft size={16} />
          Screen Another Child
        </Link>
        <Link to="/about" className="btn btn-outline-primary">
          Learn More About Methodology
        </Link>
      </div>
    </div>
  );
}
