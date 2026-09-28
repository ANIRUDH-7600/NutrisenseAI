import React from 'react';
import {
  BookOpen,
  Database,
  Cpu,
  Target,
  AlertTriangle,
  Layers
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function AboutPage() {
  return (
    <div className="container" style={{ paddingTop: '2rem', maxWidth: '960px' }}>
      <DisclaimerBanner />

      {/* Page Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.3rem 0.75rem',
          backgroundColor: 'var(--color-primary-light)',
          color: 'var(--color-primary)',
          borderRadius: 'var(--radius-full)',
          fontSize: '0.825rem',
          fontWeight: 700,
          marginBottom: '0.75rem',
          maxWidth: '100%'
        }}>
          <BookOpen size={15} style={{ flexShrink: 0 }} />
          <span>Research Methodology & System Architecture</span>
        </div>
        <h1 style={{ marginBottom: '0.5rem' }}>Methodology & Scientific Foundations</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1rem', lineHeight: '1.6' }}>
          NutriSense AI is an academic machine learning research system developed to investigate the feasibility
          of non-invasive community pre-screening for under-five undernutrition in resource-constrained environments.
        </p>
      </div>

      {/* Dataset & Population Card */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1rem' }}>
          <Database size={22} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.35rem' }}>Dataset & Target Population</h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '1.5rem', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', color: 'var(--color-primary)', marginBottom: '0.35rem' }}>
              Data Source
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Trained and cross-validated on the <strong>India National Family Health Survey (NFHS-5, 2019–21)</strong>,
              utilizing the standard Demographic and Health Surveys (DHS) Children's Recode (KR) dataset.
            </p>
          </div>
          <div>
            <h3 style={{ fontSize: '1rem', color: 'var(--color-primary)', marginBottom: '0.35rem' }}>
              Target Population
            </h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Under-five children aged <strong>0 to 59 completed months</strong>, encompassing standard demographic,
              maternal reproductive histories, infant feeding practices, and household socioeconomics.
            </p>
          </div>
        </div>
        <div style={{
          padding: '0.85rem 1rem',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: 'var(--bg-subtle)',
          fontSize: '0.825rem',
          color: 'var(--text-muted)'
        }}>
          <strong>Privacy Notice:</strong> Raw survey microdata is strictly quarantined
          for offline training and statistical validation. No microdata records are packaged, downloaded, or accessible via the application.
        </div>
      </div>

      {/* Scenario A Rationale Card */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1rem' }}>
          <Layers size={22} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.35rem' }}>Scenario A: Non-Invasive Community Triage</h2>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.6', marginBottom: '1.25rem' }}>
          In many rural and tribal Anganwadi centers across India, calibrated digital hanging scales and infantometers
          suffer from frequent calibration drift, unmeasured children due to equipment shortage, or transport barriers.
          <strong>Scenario A</strong> evaluates whether 30 non-invasive features—solicited via standard maternal interview—can
          serve as a high-sensitivity triage filter to detect children requiring formal measurement.
        </p>

        <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Strict Feature Isolation Policy</h3>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5', marginBottom: '0.75rem' }}>
          To ensure true predictive validity and avoid mathematical data leakage:
        </p>
        <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--text-main)', lineHeight: '1.6' }}>
          <li>
            <strong>Excluded Anthropometric Measurements:</strong> Direct height (<code>hw3</code>), weight (<code>hw2</code>),
            Z-scores (<code>hw70</code>, <code>hw71</code>, <code>hw72</code>, <code>hw73</code>), and anemia measurements (<code>hw57</code>)
            are strictly forbidden from the input feature set.
          </li>
          <li>
            <strong>Permitted Features:</strong> Only child demographics, birth order, subjective birth size, 2-week symptom recall (diarrhea, fever, cough),
            maternal age/BMI, JMP water/sanitation status, and household wealth quintiles are accepted.
          </li>
        </ul>
      </div>

      {/* Model & Explainability Card */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1rem' }}>
          <Cpu size={22} color="var(--color-primary)" />
          <h2 style={{ fontSize: '1.35rem' }}>Model Architecture & Explainability</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '1.5rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', marginBottom: '0.35rem' }}>Algorithm</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              <strong>LightGBM (Light Gradient Boosting Machine)</strong> was selected through multi-algorithm
              benchmarking (evaluating Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM)
              due to superior calibration, fast inference, and robust handling of mixed numerical/categorical data.
            </p>
          </div>
          <div>
            <h3 style={{ fontSize: '1rem', marginBottom: '0.35rem' }}>SHAP Explainability</h3>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Model decisions are governed by cooperative game theory through <strong>TreeSHAP</strong>. Feature importance audits
              confirm that child age, maternal BMI, wealth index, and birth interval contribute top predictive power,
              mirroring established public health literature.
            </p>
          </div>
        </div>

        <div style={{
          padding: '1rem',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'var(--bg-subtle)',
          border: '1px solid var(--border-color)'
        }}>
          <h4 style={{ fontSize: '0.9rem', marginBottom: '0.35rem' }}>Locked Operating Decision Thresholds</h4>
          <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
            Operating thresholds are calibrated from the research validation split to optimize community screening sensitivity:
          </p>
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.5rem', flexWrap: 'wrap', fontWeight: 600, fontSize: '0.85rem' }}>
            <span>Stunting: &tau; = 0.36</span>
            <span>Underweight: &tau; = 0.30</span>
            <span>Wasting: &tau; = 0.17</span>
          </div>
        </div>
      </div>

      {/* Limitations & Ethical Boundaries */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1rem' }}>
          <AlertTriangle size={22} color="var(--color-warning)" />
          <h2 style={{ fontSize: '1.35rem' }}>Scientific Boundaries & Limitations</h2>
        </div>
        <ul style={{ paddingLeft: '1.25rem', fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
          <li>
            <strong>No Causal Claims:</strong> Predictive correlations identify risk profiles but do not imply direct causality.
            Interventions should adhere to standard nutritional protocols.
          </li>
          <li>
            <strong>Non-Diagnostic:</strong> A screen-positive result does not mean the child has clinical malnutrition; it indicates
            the child warrants direct measurement with calibrated instruments.
          </li>
          <li>
            <strong>No Prescription Engine:</strong> NutriSense AI deliberately omits therapeutic feeding dosages, drug prescriptions,
            or medical treatment regimens.
          </li>
        </ul>
      </div>
    </div>
  );
}
