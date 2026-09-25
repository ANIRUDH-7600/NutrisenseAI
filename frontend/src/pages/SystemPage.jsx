import React, { useState, useEffect } from 'react';
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  RotateCw,
  Server,
  ShieldCheck,
  Layers,
  FileCode,
  Tag,
  Clock
} from 'lucide-react';
import { apiService } from '../services/api';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function SystemPage() {
  const [modelHealth, setModelHealth] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [serviceHealth, setServiceHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);

  const fetchDiagnostics = async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthRes, modelHealthRes, metadataRes] = await Promise.all([
        apiService.getHealth(),
        apiService.getModelHealth(),
        apiService.getMetadata()
      ]);

      setServiceHealth(healthRes);
      setModelHealth(modelHealthRes);
      setMetadata(metadataRes);
      setLastChecked(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  return (
    <div className="container" style={{ paddingTop: '2rem', maxWidth: '960px' }}>
      <DisclaimerBanner />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
        <div>
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
            marginBottom: '0.5rem'
          }}>
            <Cpu size={15} />
            <span>Diagnostics & Model Provenance</span>
          </div>
          <h1 style={{ fontSize: '2rem', marginBottom: '0.25rem' }}>System & Model Health</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.925rem' }}>
            Live cryptographic verification and operational readiness status of the NutriSense AI inference backend.
          </p>
        </div>

        <button
          onClick={fetchDiagnostics}
          className="btn btn-secondary btn-sm"
          disabled={loading}
          title="Query backend health endpoints"
        >
          <RotateCw size={14} className={loading ? 'animate-spin' : ''} />
          <span>{loading ? 'Refreshing...' : 'Refresh Status'}</span>
        </button>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          role="alert"
          style={{
            padding: '1.25rem',
            marginBottom: '2rem',
            backgroundColor: 'var(--color-danger-light)',
            border: '1px solid var(--color-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--color-danger)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, marginBottom: '0.35rem' }}>
            <AlertTriangle size={18} />
            <span>Service Diagnostic Error ({error.code})</span>
          </div>
          <p style={{ fontSize: '0.875rem' }}>{error.message}</p>
        </div>
      )}

      {/* Health Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem', marginBottom: '2rem' }}>
        {/* Service Liveness */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>API Service Liveness</span>
            <Server size={18} color="var(--color-primary)" />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '0.75rem', height: '0.75rem', borderRadius: '50%',
              backgroundColor: serviceHealth?.status === 'healthy' ? 'var(--color-success)' : 'var(--color-danger)'
            }} />
            <span style={{ fontSize: '1.35rem', fontWeight: 700, textTransform: 'capitalize' }}>
              {serviceHealth?.status || 'Offline'}
            </span>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', marginTop: '0.5rem', display: 'block' }}>
            Endpoint: GET /health (zero microdata access)
          </span>
        </div>

        {/* Model Integrity */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>Model Integrity Audit</span>
            <ShieldCheck size={18} color="var(--color-primary)" />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '0.75rem', height: '0.75rem', borderRadius: '50%',
              backgroundColor: modelHealth?.model_integrity_verified ? 'var(--color-success)' : 'var(--color-danger)'
            }} />
            <span style={{ fontSize: '1.35rem', fontWeight: 700 }}>
              {modelHealth?.model_integrity_verified ? 'Verified & In-Memory' : 'Unverified'}
            </span>
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', marginTop: '0.5rem', display: 'block' }}>
            Endpoint: GET /api/v1/health/model
          </span>
        </div>

        {/* Last Checked */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600 }}>Last Telemetry Poll</span>
            <Clock size={18} color="var(--color-primary)" />
          </div>
          <div style={{ fontSize: '1.35rem', fontWeight: 700 }}>
            {lastChecked || 'Pending'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-subtle)', marginTop: '0.5rem', display: 'block' }}>
            Environment: {import.meta.env.MODE || 'development'}
          </span>
        </div>
      </div>

      {/* Model Registry Specifications */}
      {metadata && (
        <div className="card" style={{ marginBottom: '2rem' }}>
          <div className="card-header">
            <h2 className="card-title">
              <Tag size={20} color="var(--color-primary)" />
              Registered Model Specifications
            </h2>
            <p className="card-subtitle">
              Public provenance records provided by the Step-17 model registry.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem' }}>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Model Release Version</span>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginTop: '0.15rem' }}>
                {metadata.model_version}
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Feature Schema Version</span>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginTop: '0.15rem' }}>
                {metadata.feature_schema_version} (30 Features)
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Model Family</span>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginTop: '0.15rem' }}>
                {metadata.model_family} (Unweighted)
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Screening Scenario</span>
              <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-main)', marginTop: '0.15rem' }}>
                Scenario {metadata.scenario} (Scale-Free Community Triage)
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-color)' }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Operating Screening Decision Boundaries</h4>
            <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
              {metadata.threshold_values &&
                Object.entries(metadata.threshold_values).map(([target, thresh]) => (
                  <div key={target} style={{
                    padding: '0.5rem 0.9rem',
                    backgroundColor: 'var(--bg-subtle)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.85rem'
                  }}>
                    <strong style={{ textTransform: 'capitalize' }}>{target}: </strong>
                    <span style={{ color: 'var(--color-primary)', fontWeight: 700 }}>
                      {(thresh * 100).toFixed(1)}% ({thresh})
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* Security & Isolation Summary Card */}
      <div className="card">
        <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Security & Privacy Guarantees</h3>
        <ul style={{ paddingLeft: '1.25rem', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
          <li>No internal filesystem paths or server directories are exposed to client applications.</li>
          <li>Raw survey microdata records are completely decoupled and quarantined.</li>
          <li>All predictions are computed synchronously in-memory by approved LightGBM estimators.</li>
          <li>Client requests are forbidden from injecting arbitrary models, thresholds, or custom feature schemas.</li>
        </ul>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}
