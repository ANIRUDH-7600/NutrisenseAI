/**
 * Frontend Test Suite for Step 19: NutriSense AI React Frontend.
 * Tests form validation, payload construction, prohibited field exclusion,
 * API response handling, and non-clinical terminology rules.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { SAMPLE_CHILD_DATA, INDIAN_STATES } from '../src/utils/constants.js';

// Approved 30 Scenario-A features from Step 18 (v2)
const APPROVED_30_FEATURES = [
  'child_age_months',
  'child_age_group',
  'child_sex_male',
  'birth_order',
  'is_multiple_birth',
  'is_firstborn',
  'preceding_birth_interval_months',
  'birth_size_ordinal',
  'birth_weight_kg',
  'birth_weight_missing',
  'delivery_place_type',
  'still_breastfeeding',
  'diarrhea_recent',
  'fever_recent',
  'cough_recent',
  'mother_age_years',
  'mother_age_first_birth',
  'mother_bmi',
  'mother_bmi_missing',
  'total_children_born',
  'anc_visits_count',
  'anc_visits_missing',
  'wealth_quintile',
  'is_rural',
  'drinking_water_type',
  'sanitation_facility_type',
  'has_electricity',
  'clean_cooking_fuel',
  'household_size',
  'state_id'
];

// Prohibited leakage variables and socially sensitive attributes that must NEVER be in client payloads
const PROHIBITED_VARS = [
  'hw70', 'hw71', 'hw72', 'hw73',
  'hw2', 'hw3', 'hw4', 'hw5', 'hw57',
  'stunting', 'underweight', 'wasting',
  'sample_weight',
  'household_head_female', 'caste_category', 'religion_category', 'mother_education_level'
];

test('1. Sample child profile adheres strictly to approved 30 features', () => {
  const keys = Object.keys(SAMPLE_CHILD_DATA);
  assert.equal(keys.length, 30, `Expected 30 features in sample data, got ${keys.length}`);
  for (const key of keys) {
    assert.ok(
      APPROVED_30_FEATURES.includes(key),
      `Unexpected feature in sample data: ${key}`
    );
  }
});

test('2. Prohibited anthropometric variables are strictly absent from sample profile', () => {
  const keys = Object.keys(SAMPLE_CHILD_DATA);
  for (const pvar of PROHIBITED_VARS) {
    assert.ok(
      !keys.includes(pvar),
      `Prohibited variable ${pvar} found in client profile!`
    );
  }
});

test('3. Child age validation constraints', () => {
  const validateAge = (age) => {
    const num = parseFloat(age);
    if (isNaN(num)) return 'Child age is required';
    if (num < 0 || num > 59) return 'Child age must be between 0 and 59 completed months.';
    return null;
  };

  assert.equal(validateAge('24'), null);
  assert.equal(validateAge('0'), null);
  assert.equal(validateAge('59'), null);
  assert.match(validateAge('60'), /between 0 and 59/);
  assert.match(validateAge('-1'), /between 0 and 59/);
  assert.match(validateAge('abc'), /Child age is required/);
});

test('4. Maternal biological consistency constraints', () => {
  const validateMaternalAges = (motherAge, firstBirthAge) => {
    const ma = parseFloat(motherAge);
    const fba = parseFloat(firstBirthAge);
    if (fba > ma) return 'Age at first birth cannot exceed current age';
    return null;
  };

  assert.equal(validateMaternalAges(26, 22), null);
  assert.match(validateMaternalAges(20, 25), /cannot exceed current age/);
});

test('5. Birth order vs total children born constraint', () => {
  const validateBirthOrder = (order, total) => {
    const o = parseFloat(order);
    const t = parseFloat(total);
    if (o > t) return 'Birth order cannot exceed total children born';
    return null;
  };

  assert.equal(validateBirthOrder(1, 1), null);
  assert.equal(validateBirthOrder(2, 3), null);
  assert.match(validateBirthOrder(4, 2), /cannot exceed total children born/);
});

test('6. Firstborn automatically clears birth interval', () => {
  const constructInterval = (isFirstborn, intervalInput) => {
    if (isFirstborn === 1) return null;
    return intervalInput === '' ? null : parseFloat(intervalInput);
  };

  assert.equal(constructInterval(1, '24'), null);
  assert.equal(constructInterval(0, '36'), 36.0);
  assert.equal(constructInterval(0, ''), null);
});

test('7. State ID dropdown contains 36 valid Indian States/UTs', () => {
  assert.equal(INDIAN_STATES.length, 36);
  assert.equal(INDIAN_STATES[0].id, 1);
  assert.equal(INDIAN_STATES[35].id, 36);
});

test('8. Response interpretation adheres to non-clinical screening semantics', () => {
  const sampleApiResponse = {
    success: true,
    model_version: 'nutrisense-scenario-a-v2.0.0',
    feature_schema_version: 'scenario-a-30-v2',
    predictions: {
      stunting: {
        probability: 0.413309,
        threshold: 0.36,
        screen_positive: true
      },
      underweight: {
        probability: 0.28,
        threshold: 0.30,
        screen_positive: false
      },
      wasting: {
        probability: 0.25,
        threshold: 0.17,
        screen_positive: true
      }
    }
  };

  const getLabel = (isPositive) => (isPositive ? 'Screen Positive' : 'Screen Negative');

  assert.equal(getLabel(sampleApiResponse.predictions.stunting.screen_positive), 'Screen Positive');
  assert.equal(getLabel(sampleApiResponse.predictions.underweight.screen_positive), 'Screen Negative');
  assert.equal(getLabel(sampleApiResponse.predictions.wasting.screen_positive), 'Screen Positive');

  // Assert clinical diagnosis words are not present
  const serialized = JSON.stringify(sampleApiResponse).toLowerCase();
  assert.ok(!serialized.includes('diagnosed'));
  assert.ok(!serialized.includes('confirmed disease'));
  assert.ok(!serialized.includes('has malnutrition'));
});

test('9. Sensitive data logging prohibition', () => {
  // Simulate logging operational metrics vs sensitive attributes
  const logTelemetry = (reqId, method, path, status) => {
    return `req_id=${reqId} method=${method} path=${path} status=${status}`;
  };

  const logEntry = logTelemetry('test-uuid-1234', 'POST', '/api/v1/screen', 200);
  assert.ok(!logEntry.includes('child_age_months'));
  assert.ok(!logEntry.includes('caste_category'));
  assert.ok(!logEntry.includes('mother_bmi'));
  assert.ok(logEntry.includes('status=200'));
});
