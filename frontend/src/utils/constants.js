/**
 * NutriSense AI Constants & Form Options
 * Strictly aligned with Step 18/22 backend schemas and NFHS-5 codebooks.
 */

export const INDIAN_STATES = [
  { id: 1, name: "Jammu & Kashmir" },
  { id: 2, name: "Himachal Pradesh" },
  { id: 3, name: "Punjab" },
  { id: 4, name: "Chandigarh" },
  { id: 5, name: "Uttarakhand" },
  { id: 6, name: "Haryana" },
  { id: 7, name: "Delhi" },
  { id: 8, name: "Rajasthan" },
  { id: 9, name: "Uttar Pradesh" },
  { id: 10, name: "Bihar" },
  { id: 11, name: "Sikkim" },
  { id: 12, name: "Arunachal Pradesh" },
  { id: 13, name: "Nagaland" },
  { id: 14, name: "Manipur" },
  { id: 15, name: "Mizoram" },
  { id: 16, name: "Tripura" },
  { id: 17, name: "Meghalaya" },
  { id: 18, name: "Assam" },
  { id: 19, name: "West Bengal" },
  { id: 20, name: "Jharkhand" },
  { id: 21, name: "Odisha" },
  { id: 22, name: "Chhattisgarh" },
  { id: 23, name: "Madhya Pradesh" },
  { id: 24, name: "Gujarat" },
  { id: 25, name: "Daman & Diu" },
  { id: 26, name: "Dadra & Nagar Haveli" },
  { id: 27, name: "Maharashtra" },
  { id: 28, name: "Andhra Pradesh" },
  { id: 29, name: "Karnataka" },
  { id: 30, name: "Goa" },
  { id: 31, name: "Lakshadweep" },
  { id: 32, name: "Kerala" },
  { id: 33, name: "Tamil Nadu" },
  { id: 34, name: "Puducherry" },
  { id: 35, name: "Andaman & Nicobar Islands" },
  { id: 36, name: "Telangana" }
];

export const DELIVERY_PLACES = [
  { value: "Public_Facility", label: "Public Health Facility (Govt Hospital / CHC / PHC)" },
  { value: "Private_Facility", label: "Private Hospital / Nursing Home / Clinic" },
  { value: "Home_Delivery", label: "Home Delivery" },
  { value: "Other", label: "Other Location" }
];

export const WATER_TYPES = [
  { value: "Improved_Piped", label: "Improved Piped Water (Tap in dwelling/yard)" },
  { value: "Improved_Groundwater_Bottled", label: "Improved Groundwater / Borewell / Bottled / RO" },
  { value: "Unimproved_Surface_Other", label: "Unimproved Well / Spring / Tanker" },
  { value: "Unimproved_Surface", label: "Surface Water (River / Pond / Canal / Lake)" }
];

export const SANITATION_TYPES = [
  { value: "Flush_Toilet", label: "Improved Flush / Pour-Flush Toilet" },
  { value: "Pit_Latrine", label: "Improved Pit Latrine with Slab" },
  { value: "Open_Defecation_None", label: "No Facility / Bush / Open Defecation" },
  { value: "Other_Facility", label: "Shared / Other Unimproved Facility" }
];

export const BIRTH_SIZE_OPTIONS = [
  { value: 1.0, label: "1 — Very Large" },
  { value: 2.0, label: "2 — Larger than Average" },
  { value: 3.0, label: "3 — Average Size" },
  { value: 4.0, label: "4 — Smaller than Average" },
  { value: 5.0, label: "5 — Very Small" }
];

export const WEALTH_QUINTILES = [
  { value: 1.0, label: "Quintile 1 — Poorest (Lowest 20%)" },
  { value: 2.0, label: "Quintile 2 — Poorer (Second 20%)" },
  { value: 3.0, label: "Quintile 3 — Middle (Middle 20%)" },
  { value: 4.0, label: "Quintile 4 — Richer (Fourth 20%)" },
  { value: 5.0, label: "Quintile 5 — Richest (Top 20%)" }
];

/**
 * Realistic, fully valid sample child profile for testing and rapid demonstration (v2 30-feature schema).
 * Sensitive social attributes (mother_education_level, caste_category, religion_category, household_head_female) excluded.
 */
export const SAMPLE_CHILD_DATA = {
  child_age_months: 24.0,
  child_age_group: "24_35_mo",
  child_sex_male: 1,
  birth_order: 2.0,
  is_multiple_birth: 0,
  is_firstborn: 0,
  preceding_birth_interval_months: 28.0,
  birth_size_ordinal: 3.0,
  birth_weight_kg: 2.8,
  birth_weight_missing: 0,
  delivery_place_type: "Public_Facility",
  still_breastfeeding: 1.0,
  diarrhea_recent: 0.0,
  fever_recent: 0.0,
  cough_recent: 0.0,
  mother_age_years: 26.0,
  mother_age_first_birth: 22.0,
  mother_bmi: 21.5,
  mother_bmi_missing: 0,
  total_children_born: 2.0,
  anc_visits_count: 4.0,
  anc_visits_missing: 0,
  wealth_quintile: 2.0,
  is_rural: 1,
  drinking_water_type: "Improved_Piped",
  sanitation_facility_type: "Pit_Latrine",
  has_electricity: 1,
  clean_cooking_fuel: 1,
  household_size: 5.0,
  state_id: 10
};
