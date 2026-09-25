import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function DisclaimerBanner({ text, compact = false }) {
  const defaultText =
    "This tool provides an AI-based community pre-screening result and is not a medical diagnosis. Screening results should be interpreted by an appropriately qualified healthcare professional.";

  return (
    <div className="disclaimer-banner" role="alert">
      <AlertTriangle className="disclaimer-banner-icon" size={compact ? 18 : 22} />
      <div>
        <strong style={{ display: 'block', marginBottom: '0.15rem' }}>
          Research Screening System — Non-Diagnostic Tool
        </strong>
        <span>{text || defaultText}</span>
      </div>
    </div>
  );
}
