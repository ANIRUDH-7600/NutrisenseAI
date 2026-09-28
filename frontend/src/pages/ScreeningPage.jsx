import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  Send,
  AlertCircle,
  RotateCcw
} from 'lucide-react';
import { apiService } from '../services/api';
import DisclaimerBanner from '../components/DisclaimerBanner';
import {
  INDIAN_STATES,
  DELIVERY_PLACES,
  WATER_TYPES,
  SANITATION_TYPES,
  BIRTH_SIZE_OPTIONS,
  WEALTH_QUINTILES,
  SAMPLE_CHILD_DATA
} from '../utils/constants';

const INITIAL_FORM = {
  // Identification
  child_name: '',
  // A. Child Demographics
  child_age_months: '',
  child_sex_male: '1',
  birth_order: '1',
  is_multiple_birth: '0',
  preceding_birth_interval_months: '',
  is_firstborn: '1',
  birth_size_ordinal: '3.0',
  birth_weight_kg: '',
  // B. Morbidities
  diarrhea_recent: '0',
  fever_recent: '0',
  cough_recent: '0',
  // C. Feeding & Delivery
  delivery_place_type: 'Public_Facility',
  still_breastfeeding: '1',
  // D. Maternal
  mother_age_years: '',
  mother_age_first_birth: '',
  mother_bmi: '',
  total_children_born: '1',
  anc_visits_count: '',
  // E. Socioeconomic
  wealth_quintile: '2.0',
  is_rural: '1',
  household_size: '5',
  // F. Environment
  drinking_water_type: 'Improved_Piped',
  sanitation_facility_type: 'Pit_Latrine',
  has_electricity: '1',
  clean_cooking_fuel: '1',
  // G. Location
  state_id: '10'
};

export default function ScreeningPage({ onScreeningSuccess }) {
  const navigate = useNavigate();
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiError, setApiError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => {
      const next = { ...prev, [name]: value };

      // Automatic firstborn logic
      if (name === 'birth_order') {
        const order = parseFloat(value);
        if (order === 1) {
          next.is_firstborn = '1';
          next.preceding_birth_interval_months = '';
        } else if (order > 1) {
          next.is_firstborn = '0';
        }
      }
      return next;
    });

    // Clear field-level error on change
    if (errors[name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  const handleFillSample = () => {
    const sampleFormatted = { child_name: 'Aarav Sharma' };
    Object.keys(INITIAL_FORM).forEach((key) => {
      if (key === 'child_name') return;
      const val = SAMPLE_CHILD_DATA[key];
      sampleFormatted[key] = val !== null && val !== undefined ? String(val) : '';
    });
    setFormData(sampleFormatted);
    setErrors({});
    setApiError(null);
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM);
    setErrors({});
    setApiError(null);
  };

  const validate = () => {
    const errs = {};

    // 1. Child Demographics
    const age = parseFloat(formData.child_age_months);
    if (formData.child_age_months === '' || isNaN(age)) {
      errs.child_age_months = 'Child age in months is required.';
    } else if (age < 0 || age > 59) {
      errs.child_age_months = 'Child age must be between 0 and 59 completed months.';
    }

    const birthOrder = parseFloat(formData.birth_order);
    if (formData.birth_order === '' || isNaN(birthOrder) || birthOrder < 1) {
      errs.birth_order = 'Birth order must be an integer >= 1.';
    }

    if (formData.birth_weight_kg !== '') {
      const bw = parseFloat(formData.birth_weight_kg);
      if (isNaN(bw) || bw < 0.5 || bw > 7.0) {
        errs.birth_weight_kg = 'Birth weight must be between 0.5 kg and 7.0 kg.';
      }
    }

    // 2. Maternal
    const motherAge = parseFloat(formData.mother_age_years);
    if (formData.mother_age_years === '' || isNaN(motherAge)) {
      errs.mother_age_years = "Mother's current age is required.";
    } else if (motherAge < 12 || motherAge > 55) {
      errs.mother_age_years = "Mother's age must be between 12 and 55 years.";
    }

    const firstBirthAge = parseFloat(formData.mother_age_first_birth);
    if (formData.mother_age_first_birth === '' || isNaN(firstBirthAge)) {
      errs.mother_age_first_birth = 'Age at first birth is required.';
    } else if (firstBirthAge < 10 || firstBirthAge > 50) {
      errs.mother_age_first_birth = 'Age at first birth must be between 10 and 50 years.';
    } else if (!isNaN(motherAge) && firstBirthAge > motherAge) {
      errs.mother_age_first_birth = `Age at first birth (${firstBirthAge}) cannot exceed current age (${motherAge}).`;
    }

    const totalChildren = parseFloat(formData.total_children_born);
    if (formData.total_children_born === '' || isNaN(totalChildren) || totalChildren < 1) {
      errs.total_children_born = 'Total children born must be >= 1.';
    } else if (!isNaN(birthOrder) && birthOrder > totalChildren) {
      errs.birth_order = `Birth order (${birthOrder}) cannot exceed total children born (${totalChildren}).`;
    }

    if (formData.mother_bmi !== '') {
      const bmi = parseFloat(formData.mother_bmi);
      if (isNaN(bmi) || bmi < 10.0 || bmi > 60.0) {
        errs.mother_bmi = 'Maternal BMI must be in plausible range [10.0, 60.0].';
      }
    }

    // 3. Household
    const hhSize = parseFloat(formData.household_size);
    if (formData.household_size === '' || isNaN(hhSize) || hhSize < 1) {
      errs.household_size = 'Household size must be at least 1 person.';
    }

    return errs;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setApiError(null);

    const clientErrors = validate();
    if (Object.keys(clientErrors).length > 0) {
      setErrors(clientErrors);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }

    // Prepare strictly compliant 30-feature payload for Scenario-A v2 API
    const isFirstbornInt = parseInt(formData.is_firstborn, 10);
    const payload = {
      // Identification
      child_name: (formData.child_name || '').trim() || 'Anonymous Child',
      // 1. Child
      child_age_months: parseFloat(formData.child_age_months),
      child_sex_male: parseInt(formData.child_sex_male, 10),
      birth_order: parseFloat(formData.birth_order),
      is_multiple_birth: parseInt(formData.is_multiple_birth, 10),
      is_firstborn: isFirstbornInt,
      preceding_birth_interval_months:
        isFirstbornInt === 1 || formData.preceding_birth_interval_months === ''
          ? null
          : parseFloat(formData.preceding_birth_interval_months),
      birth_size_ordinal: parseFloat(formData.birth_size_ordinal),
      birth_weight_kg: formData.birth_weight_kg !== '' ? parseFloat(formData.birth_weight_kg) : null,
      birth_weight_missing: formData.birth_weight_kg === '' ? 1 : 0,
      delivery_place_type: formData.delivery_place_type,
      // 2. Feeding & Morbidities
      still_breastfeeding: parseFloat(formData.still_breastfeeding),
      diarrhea_recent: parseFloat(formData.diarrhea_recent),
      fever_recent: parseFloat(formData.fever_recent),
      cough_recent: parseFloat(formData.cough_recent),
      // 3. Maternal
      mother_age_years: parseFloat(formData.mother_age_years),
      mother_age_first_birth: parseFloat(formData.mother_age_first_birth),
      mother_bmi: formData.mother_bmi !== '' ? parseFloat(formData.mother_bmi) : null,
      mother_bmi_missing: formData.mother_bmi === '' ? 1 : 0,
      total_children_born: parseFloat(formData.total_children_born),
      anc_visits_count: formData.anc_visits_count !== '' ? parseFloat(formData.anc_visits_count) : null,
      anc_visits_missing: formData.anc_visits_count === '' ? 1 : 0,
      // 4. Socioeconomics & Environment
      wealth_quintile: parseFloat(formData.wealth_quintile),
      is_rural: parseFloat(formData.is_rural),
      drinking_water_type: formData.drinking_water_type,
      sanitation_facility_type: formData.sanitation_facility_type,
      has_electricity: parseFloat(formData.has_electricity),
      clean_cooking_fuel: parseFloat(formData.clean_cooking_fuel),
      household_size: parseFloat(formData.household_size),
      state_id: parseInt(formData.state_id, 10)
    };

    setIsSubmitting(true);
    try {
      const response = await apiService.screenChild(payload);
      if (onScreeningSuccess) {
        onScreeningSuccess(response, payload);
      }
      navigate('/results', { state: { result: response, inputPayload: payload } });
    } catch (err) {
      setApiError(err);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container" style={{ paddingTop: '2rem', maxWidth: '960px' }}>
      <DisclaimerBanner />

      {/* Header & Demo Action */}
      <div className="page-header-row">
        <div>
          <h1 style={{ marginBottom: '0.25rem' }}>Scenario A: Community Child Screening</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.925rem' }}>
            Enter non-invasive child, maternal, and household attributes for rapid undernutrition triage.
          </p>
        </div>
        <div className="action-btn-group">
          <button
            type="button"
            onClick={handleFillSample}
            className="btn btn-outline-primary btn-sm"
            title="Populate realistic sample data for quick demonstration"
          >
            <Sparkles size={14} />
            Fill Sample Data
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="btn btn-secondary btn-sm"
            title="Clear all fields"
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </div>

      {/* API Error Notification */}
      {apiError && (
        <div
          role="alert"
          style={{
            padding: '1.25rem',
            marginBottom: '1.5rem',
            backgroundColor: 'var(--color-danger-light)',
            border: '1px solid var(--color-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--color-danger)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, marginBottom: '0.35rem' }}>
            <AlertCircle size={18} />
            <span>Screening Request Failed ({apiError.code})</span>
          </div>
          <p style={{ fontSize: '0.875rem', marginBottom: apiError.details?.length ? '0.5rem' : '0' }}>
            {apiError.message}
          </p>
          {apiError.details && apiError.details.length > 0 && (
            <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem' }}>
              {apiError.details.map((d, idx) => (
                <li key={idx}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Client Validation Summary */}
      {Object.keys(errors).length > 0 && (
        <div
          role="alert"
          style={{
            padding: '1rem',
            marginBottom: '1.5rem',
            backgroundColor: 'var(--color-danger-light)',
            border: '1px solid var(--color-danger-border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--color-danger)',
            fontSize: '0.875rem'
          }}
        >
          <strong>Please resolve the highlighted fields before submitting:</strong>
          <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem' }}>
            {Object.values(errors).map((msg, i) => (
              <li key={i}>{msg}</li>
            ))}
          </ul>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        {/* Section A: Child Information */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">A</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Child Demographics & Birth Characteristics</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Age, biological sex, birth order, and birth size</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group" style={{ gridColumn: 'span 2' }}>
              <label className="form-label" htmlFor="child_name">
                Child Name / Identifier (Optional)
              </label>
              <input
                id="child_name"
                name="child_name"
                type="text"
                maxLength={100}
                className="form-input"
                value={formData.child_name || ''}
                onChange={handleChange}
                placeholder="e.g. Aarav Sharma"
              />
              <span className="form-helper">Recorded in MongoDB screening history for health worker follow-up</span>
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="child_age_months">
                Child Age (Completed Months)
              </label>
              <input
                id="child_age_months"
                name="child_age_months"
                type="number"
                step="0.1"
                min="0"
                max="59"
                className={`form-input ${errors.child_age_months ? 'has-error' : ''}`}
                value={formData.child_age_months}
                onChange={handleChange}
                placeholder="e.g. 24"
                required
              />
              <span className="form-helper">Age bracket 0–59 months</span>
              {errors.child_age_months && <span className="form-error">{errors.child_age_months}</span>}
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="child_sex_male">
                Biological Sex
              </label>
              <select
                id="child_sex_male"
                name="child_sex_male"
                className="form-select"
                value={formData.child_sex_male}
                onChange={handleChange}
              >
                <option value="1">Male</option>
                <option value="0">Female</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="birth_order">
                Birth Order
              </label>
              <input
                id="birth_order"
                name="birth_order"
                type="number"
                min="1"
                className={`form-input ${errors.birth_order ? 'has-error' : ''}`}
                value={formData.birth_order}
                onChange={handleChange}
                placeholder="e.g. 1"
                required
              />
              <span className="form-helper">Order relative to mother's births (&gt;= 1)</span>
              {errors.birth_order && <span className="form-error">{errors.birth_order}</span>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="is_multiple_birth">
                Multiple Birth Status
              </label>
              <select
                id="is_multiple_birth"
                name="is_multiple_birth"
                className="form-select"
                value={formData.is_multiple_birth}
                onChange={handleChange}
              >
                <option value="0">Single Birth</option>
                <option value="1">Multiple Birth (Twin / Triplet)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="preceding_birth_interval_months">
                Preceding Birth Interval (Months)
              </label>
              <input
                id="preceding_birth_interval_months"
                name="preceding_birth_interval_months"
                type="number"
                min="0"
                className="form-input"
                value={formData.preceding_birth_interval_months}
                onChange={handleChange}
                disabled={formData.is_firstborn === '1'}
                placeholder={formData.is_firstborn === '1' ? 'N/A (Firstborn)' : 'e.g. 28'}
              />
              <span className="form-helper">Leave blank if child is firstborn</span>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="birth_size_ordinal">
                Mother's Assessment of Birth Size
              </label>
              <select
                id="birth_size_ordinal"
                name="birth_size_ordinal"
                className="form-select"
                value={formData.birth_size_ordinal}
                onChange={handleChange}
              >
                {BIRTH_SIZE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="birth_weight_kg">
                Recorded Birth Weight (kg)
              </label>
              <input
                id="birth_weight_kg"
                name="birth_weight_kg"
                type="number"
                step="0.05"
                min="0.5"
                max="7.0"
                className={`form-input ${errors.birth_weight_kg ? 'has-error' : ''}`}
                value={formData.birth_weight_kg}
                onChange={handleChange}
                placeholder="e.g. 2.8 (Optional if unrecorded)"
              />
              <span className="form-helper">Leave blank if unmeasured/unknown at birth</span>
              {errors.birth_weight_kg && <span className="form-error">{errors.birth_weight_kg}</span>}
            </div>
          </div>
        </div>

        {/* Section B: Recent Morbidities */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">B</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Recent Pediatric Morbidities</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Symptoms occurring within the preceding 2 weeks</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="diarrhea_recent">
                Recent Diarrhea Episode
              </label>
              <select
                id="diarrhea_recent"
                name="diarrhea_recent"
                className="form-select"
                value={formData.diarrhea_recent}
                onChange={handleChange}
              >
                <option value="0">No Diarrhea</option>
                <option value="1">Yes, Episode in Past 2 Weeks</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="fever_recent">
                Recent Fever Episode
              </label>
              <select
                id="fever_recent"
                name="fever_recent"
                className="form-select"
                value={formData.fever_recent}
                onChange={handleChange}
              >
                <option value="0">No Fever</option>
                <option value="1">Yes, Episode in Past 2 Weeks</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="cough_recent">
                Recent Cough Episode
              </label>
              <select
                id="cough_recent"
                name="cough_recent"
                className="form-select"
                value={formData.cough_recent}
                onChange={handleChange}
              >
                <option value="0">No Cough</option>
                <option value="1">Yes, Episode in Past 2 Weeks</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section C: Feeding & Delivery */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">C</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Feeding Practices & Delivery Setting</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Breastfeeding continuation and birth place</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="still_breastfeeding">
                Currently Breastfeeding
              </label>
              <select
                id="still_breastfeeding"
                name="still_breastfeeding"
                className="form-select"
                value={formData.still_breastfeeding}
                onChange={handleChange}
              >
                <option value="1">Yes, Currently Breastfeeding</option>
                <option value="0">No, Stopped or Never Breastfed</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="delivery_place_type">
                Delivery Location
              </label>
              <select
                id="delivery_place_type"
                name="delivery_place_type"
                className="form-select"
                value={formData.delivery_place_type}
                onChange={handleChange}
              >
                {DELIVERY_PLACES.map((dp) => (
                  <option key={dp.value} value={dp.value}>
                    {dp.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Section D: Maternal Characteristics */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">D</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Maternal Characteristics & History</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Age, parity, maternal BMI, and antenatal care</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="mother_age_years">
                Mother's Age (Years)
              </label>
              <input
                id="mother_age_years"
                name="mother_age_years"
                type="number"
                min="12"
                max="55"
                className={`form-input ${errors.mother_age_years ? 'has-error' : ''}`}
                value={formData.mother_age_years}
                onChange={handleChange}
                placeholder="e.g. 26"
                required
              />
              {errors.mother_age_years && <span className="form-error">{errors.mother_age_years}</span>}
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="mother_age_first_birth">
                Mother's Age at First Live Birth
              </label>
              <input
                id="mother_age_first_birth"
                name="mother_age_first_birth"
                type="number"
                min="10"
                max="50"
                className={`form-input ${errors.mother_age_first_birth ? 'has-error' : ''}`}
                value={formData.mother_age_first_birth}
                onChange={handleChange}
                placeholder="e.g. 22"
                required
              />
              <span className="form-helper">Cannot exceed current mother age</span>
              {errors.mother_age_first_birth && (
                <span className="form-error">{errors.mother_age_first_birth}</span>
              )}
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="total_children_born">
                Total Children Ever Born
              </label>
              <input
                id="total_children_born"
                name="total_children_born"
                type="number"
                min="1"
                className={`form-input ${errors.total_children_born ? 'has-error' : ''}`}
                value={formData.total_children_born}
                onChange={handleChange}
                placeholder="e.g. 2"
                required
              />
              <span className="form-helper">Must be &gt;= birth order</span>
              {errors.total_children_born && <span className="form-error">{errors.total_children_born}</span>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="mother_bmi">
                Mother's BMI (kg/m²)
              </label>
              <input
                id="mother_bmi"
                name="mother_bmi"
                type="number"
                step="0.1"
                min="10"
                max="60"
                className={`form-input ${errors.mother_bmi ? 'has-error' : ''}`}
                value={formData.mother_bmi}
                onChange={handleChange}
                placeholder="e.g. 21.5 (Optional)"
              />
              <span className="form-helper">Leave blank if unmeasured</span>
              {errors.mother_bmi && <span className="form-error">{errors.mother_bmi}</span>}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="anc_visits_count">
                Antenatal Care (ANC) Visits Count
              </label>
              <input
                id="anc_visits_count"
                name="anc_visits_count"
                type="number"
                min="0"
                className="form-input"
                value={formData.anc_visits_count}
                onChange={handleChange}
                placeholder="e.g. 4 (Optional)"
              />
              <span className="form-helper">Leave blank if unknown/unrecorded</span>
            </div>
          </div>
        </div>

        {/* Section E: Household Socioeconomics */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">E</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Household Socioeconomics</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Wealth quintile, residence, and household size</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label" htmlFor="wealth_quintile">
                Household Wealth Quintile
              </label>
              <select
                id="wealth_quintile"
                name="wealth_quintile"
                className="form-select"
                value={formData.wealth_quintile}
                onChange={handleChange}
              >
                {WEALTH_QUINTILES.map((wq) => (
                  <option key={wq.value} value={wq.value}>
                    {wq.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="is_rural">
                Residence Type
              </label>
              <select
                id="is_rural"
                name="is_rural"
                className="form-select"
                value={formData.is_rural}
                onChange={handleChange}
              >
                <option value="1">Rural</option>
                <option value="0">Urban</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="household_size">
                Total Household Size
              </label>
              <input
                id="household_size"
                name="household_size"
                type="number"
                min="1"
                className={`form-input ${errors.household_size ? 'has-error' : ''}`}
                value={formData.household_size}
                onChange={handleChange}
                placeholder="e.g. 5"
                required
              />
              <span className="form-helper">Total usual household members</span>
              {errors.household_size && <span className="form-error">{errors.household_size}</span>}
            </div>
          </div>
        </div>

        {/* Section F: Household Environment */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">F</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>WASH & Living Environment</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Water, sanitation, electricity, and clean cooking fuel</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label" htmlFor="drinking_water_type">
                Drinking Water Source (JMP Standard)
              </label>
              <select
                id="drinking_water_type"
                name="drinking_water_type"
                className="form-select"
                value={formData.drinking_water_type}
                onChange={handleChange}
              >
                {WATER_TYPES.map((w) => (
                  <option key={w.value} value={w.value}>
                    {w.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="sanitation_facility_type">
                Sanitation Facility (JMP Standard)
              </label>
              <select
                id="sanitation_facility_type"
                name="sanitation_facility_type"
                className="form-select"
                value={formData.sanitation_facility_type}
                onChange={handleChange}
              >
                {SANITATION_TYPES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="has_electricity">
                Electricity Connection
              </label>
              <select
                id="has_electricity"
                name="has_electricity"
                className="form-select"
                value={formData.has_electricity}
                onChange={handleChange}
              >
                <option value="1">Yes, Connected to Electricity</option>
                <option value="0">No Electricity</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="clean_cooking_fuel">
                Cooking Fuel Type
              </label>
              <select
                id="clean_cooking_fuel"
                name="clean_cooking_fuel"
                className="form-select"
                value={formData.clean_cooking_fuel}
                onChange={handleChange}
              >
                <option value="1">Clean Fuel (LPG / Natural Gas / Electricity / Biogas)</option>
                <option value="0">Solid Biomass (Firewood / Coal / Dung Cakes / Kerosene)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section G: Geographic Location */}
        <div className="form-section-card">
          <div className="form-section-header">
            <div className="form-section-badge">G</div>
            <div>
              <h2 style={{ fontSize: '1.15rem' }}>Geographic State / Union Territory</h2>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Administrative State ID for regional contextual baseline</span>
            </div>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label form-label-required" htmlFor="state_id">
                State / Union Territory
              </label>
              <select
                id="state_id"
                name="state_id"
                className="form-select"
                value={formData.state_id}
                onChange={handleChange}
              >
                {INDIAN_STATES.map((st) => (
                  <option key={st.id} value={st.id}>
                    {st.id} — {st.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Submit Actions */}
        <div className="form-submit-row">
          <button
            type="button"
            onClick={handleReset}
            className="btn btn-secondary btn-lg"
            disabled={isSubmitting}
          >
            Clear Form
          </button>
          <button
            type="submit"
            className="btn btn-primary btn-lg"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <span className="spinner" style={{ display: 'inline-block', width: '16px', height: '16px', border: '2px solid #05070a', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                <span>Running Screening...</span>
              </>
            ) : (
              <>
                <span>Evaluate Child Undernutrition Risk</span>
                <Send size={18} />
              </>
            )}
          </button>
        </div>
      </form>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
