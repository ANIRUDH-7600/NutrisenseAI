"""
Pydantic schemas for request validation and response serialization.
NutriSense AI Scenario A Community Pre-Screening API (v2.0.0, 30 features).
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

# Excluded anthropometric, clinical, survey leakage variables, and sensitive social attributes
PROHIBITED_VARS = {
    "hw70", "hw71", "hw72", "hw73",
    "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
    "hw2", "hw3", "hw2_clean", "hw3_clean",
    "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12",
    "hw13", "hw57", "v457",
    "stunting", "underweight", "wasting",
    "stunting_severe", "underweight_severe", "wasting_severe",
    "v001", "v002", "v003", "v005", "v021", "v022", "sample_weight",
    # Four excluded sensitive social attributes
    "household_head_female", "caste_category", "religion_category", "mother_education_level"
}

ALLOWED_AGE_GROUPS = {"00_05_mo", "06_11_mo", "12_23_mo", "24_35_mo", "36_47_mo", "48_59_mo"}
ALLOWED_DELIVERY_PLACES = {"Home_Delivery", "Public_Facility", "Private_Facility", "Other"}
ALLOWED_WATER_TYPES = {"Improved_Piped", "Improved_Groundwater_Bottled", "Unimproved_Surface_Other", "Unimproved_Surface"}
ALLOWED_TOILET_TYPES = {"Flush_Toilet", "Pit_Latrine", "Open_Defecation_None", "Other_Facility"}


class ChildScreeningRequest(BaseModel):
    """
    Scenario A v2 Community Pre-Screening Child Profile Request (30 Features).
    Accepts exclusively non-invasive demographic, maternal, household, and morbidity features.
    Strictly forbids sensitive social attributes, extra unapproved fields, and anthropometric leakage.
    """
    model_config = ConfigDict(extra="forbid")

    # 1. Child Demographics
    child_age_months: float = Field(..., ge=0, le=59, description="Child's exact age in completed months (0 to 59).")
    child_age_group: Optional[str] = Field(None, description="Optional clinical age bracket (e.g. '12_23_mo'). Derived if omitted.")
    child_sex_male: int = Field(..., ge=0, le=1, description="Biological sex (1=Male, 0=Female).")
    birth_order: float = Field(..., ge=1, description="Birth order relative to mother's full birth history (>= 1).")
    is_multiple_birth: Optional[int] = Field(0, ge=0, le=1, description="Twin/multiple birth indicator (1=Multiple, 0=Single).")
    is_firstborn: Optional[int] = Field(None, ge=0, le=1, description="Firstborn indicator (1=Firstborn, 0=Later born). Derived if omitted.")
    preceding_birth_interval_months: Optional[float] = Field(None, ge=0, description="Months since previous live birth (None/NaN for firstborn).")

    # 2. Birth Characteristics
    birth_size_ordinal: Optional[float] = Field(None, ge=1, le=5, description="Mother's subjective birth size rating (1=Very large to 5=Very small).")
    birth_weight_kg: Optional[float] = Field(None, ge=0.5, le=7.0, description="Birth weight in kg (None/NaN if unrecorded/missing).")
    birth_weight_missing: Optional[int] = Field(None, ge=0, le=1, description="Birth weight missingness flag (1=Missing, 0=Recorded). Derived if omitted.")
    delivery_place_type: Optional[str] = Field("Other", description="Delivery location ('Home_Delivery', 'Public_Facility', 'Private_Facility', 'Other').")

    # 3. Feeding & Recent Morbidities
    still_breastfeeding: float = Field(..., ge=0, le=1, description="Child is currently breastfeeding (1=Yes, 0=No).")
    diarrhea_recent: float = Field(..., ge=0, le=1, description="Recent diarrhea episode in past 2 weeks (1=Yes, 0=No).")
    fever_recent: float = Field(..., ge=0, le=1, description="Recent fever episode in past 2 weeks (1=Yes, 0=No).")
    cough_recent: float = Field(..., ge=0, le=1, description="Recent cough episode in past 2 weeks (1=Yes, 0=No).")

    # 4. Maternal Characteristics
    mother_age_years: float = Field(..., ge=12, le=55, description="Mother's age at interview in completed years (12 to 55).")
    mother_age_first_birth: float = Field(..., ge=10, le=50, description="Mother's age at first live birth in years (10 to 50).")
    mother_bmi: Optional[float] = Field(None, ge=10.0, le=60.0, description="Mother's body mass index in kg/m^2 (None/NaN if unmeasured).")
    mother_bmi_missing: Optional[int] = Field(None, ge=0, le=1, description="Maternal BMI missingness flag (1=Missing, 0=Recorded). Derived if omitted.")
    total_children_born: float = Field(..., ge=1, description="Total number of children ever born to mother (>= 1).")
    anc_visits_count: Optional[float] = Field(None, ge=0, description="Number of antenatal care visits during pregnancy (None/NaN if missing).")
    anc_visits_missing: Optional[int] = Field(None, ge=0, le=1, description="ANC visits missingness flag (1=Missing, 0=Recorded). Derived if omitted.")

    # 5. Household Socioeconomics & Environment
    wealth_quintile: float = Field(..., ge=1, le=5, description="Household wealth index quintile (1=Poorest to 5=Richest).")
    is_rural: float = Field(..., ge=0, le=1, description="Place of residence (1=Rural, 0=Urban).")
    drinking_water_type: Optional[str] = Field("Improved_Piped", description="JMP water category ('Improved_Piped', 'Improved_Groundwater_Bottled', 'Unimproved_Surface_Other', 'Unimproved_Surface').")
    sanitation_facility_type: Optional[str] = Field("Flush_Toilet", description="JMP sanitation category ('Flush_Toilet', 'Pit_Latrine', 'Open_Defecation_None', 'Other_Facility').")
    has_electricity: float = Field(..., ge=0, le=1, description="Household has electricity connection (1=Yes, 0=No).")
    clean_cooking_fuel: float = Field(..., ge=0, le=1, description="Household uses clean cooking fuel (1=Yes [LPG/Electricity/Biogas], 0=Solid/Biomass).")
    household_size: float = Field(..., ge=1, description="Total number of usual household members (>= 1).")
    state_id: Union[int, str] = Field(..., description="Administrative State / Union Territory identifier (1 to 36).")

    @field_validator("child_age_group")
    @classmethod
    def validate_age_group(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_AGE_GROUPS:
            raise ValueError(f"child_age_group '{v}' is invalid. Allowed: {sorted(list(ALLOWED_AGE_GROUPS))}.")
        return v

    @field_validator("delivery_place_type")
    @classmethod
    def validate_delivery_place(cls, v: Optional[str]) -> str:
        if v is None:
            return "Other"
        norm_map = {
            "home": "Home_Delivery", "home_delivery": "Home_Delivery",
            "public": "Public_Facility", "public_facility": "Public_Facility",
            "health_facility_public": "Public_Facility",
            "private": "Private_Facility", "private_facility": "Private_Facility",
            "health_facility_private": "Private_Facility", "other": "Other"
        }
        v_norm = norm_map.get(str(v).lower(), str(v))
        if v_norm not in ALLOWED_DELIVERY_PLACES:
            raise ValueError(f"delivery_place_type '{v}' is invalid. Allowed: {sorted(list(ALLOWED_DELIVERY_PLACES))}.")
        return v_norm

    @field_validator("drinking_water_type")
    @classmethod
    def validate_water(cls, v: Optional[str]) -> str:
        if v is None:
            return "Improved_Piped"
        norm_map = {
            "improved_piped": "Improved_Piped",
            "piped": "Improved_Piped",
            "improved_groundwater_bottled": "Improved_Groundwater_Bottled",
            "groundwater": "Improved_Groundwater_Bottled",
            "unimproved_surface_other": "Unimproved_Surface_Other",
            "unimproved_surface": "Unimproved_Surface",
            "surface": "Unimproved_Surface"
        }
        v_norm = norm_map.get(str(v).lower(), str(v))
        if v_norm not in ALLOWED_WATER_TYPES:
            raise ValueError(f"drinking_water_type '{v}' is invalid. Allowed: {sorted(list(ALLOWED_WATER_TYPES))}.")
        return v_norm

    @field_validator("sanitation_facility_type")
    @classmethod
    def validate_sanitation(cls, v: Optional[str]) -> str:
        if v is None:
            return "Flush_Toilet"
        norm_map = {
            "flush": "Flush_Toilet", "flush_toilet": "Flush_Toilet",
            "pit": "Pit_Latrine", "pit_latrine": "Pit_Latrine",
            "open": "Open_Defecation_None", "open_defecation": "Open_Defecation_None",
            "open_defecation_none": "Open_Defecation_None",
            "other": "Other_Facility", "other_facility": "Other_Facility"
        }
        v_norm = norm_map.get(str(v).lower(), str(v))
        if v_norm not in ALLOWED_TOILET_TYPES:
            raise ValueError(f"sanitation_facility_type '{v}' is invalid. Allowed: {sorted(list(ALLOWED_TOILET_TYPES))}.")
        return v_norm

    @field_validator("state_id")
    @classmethod
    def validate_state(cls, v: Union[int, str]) -> int:
        try:
            sid = int(v)
            if not (1 <= sid <= 36):
                raise ValueError(f"state_id must be an administrative integer between 1 and 36, got {sid}.")
            return sid
        except (ValueError, TypeError) as e:
            raise ValueError(f"state_id must be an administrative integer between 1 and 36, got {v}.") from e

    @model_validator(mode="before")
    @classmethod
    def check_prohibited_variables(cls, data: Any) -> Any:
        """Strictly rejects any prohibited anthropometric, survey, or sensitive social variables."""
        if isinstance(data, dict):
            found_prohibited = set(data.keys()).intersection(PROHIBITED_VARS)
            if found_prohibited:
                raise ValueError(
                    f"Prohibited variable(s) supplied: {sorted(list(found_prohibited))}. "
                    "Direct anthropometric measurements, survey design variables, and sensitive social attributes "
                    "are strictly forbidden in Scenario A v2."
                )
        return data

    @model_validator(mode="after")
    def check_logical_consistency(self) -> "ChildScreeningRequest":
        """Validates biological and logical consistency."""
        if self.mother_age_first_birth > self.mother_age_years:
            raise ValueError(
                f"mother_age_first_birth ({self.mother_age_first_birth}) cannot exceed mother_age_years ({self.mother_age_years})."
            )
        if self.birth_order > self.total_children_born:
            raise ValueError(
                f"birth_order ({self.birth_order}) cannot exceed total_children_born ({self.total_children_born})."
            )
        return self


class TargetScreeningPrediction(BaseModel):
    """Screening prediction for an individual undernutrition target."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Unrounded predicted risk probability.")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Pre-specified locked decision threshold.")
    screen_positive: bool = Field(..., description="Screening outcome: True if probability >= threshold, else False.")
    decision_rule: Optional[str] = Field("probability >= threshold", description="Deterministic decision boundary rule.")
    model_family: Optional[str] = Field("LightGBM", description="Underlying model family.")
    condition: Optional[str] = Field("unweighted", description="Model training class weighting condition.")


class ScreeningResponse(BaseModel):
    """Structured response for child screening prediction."""
    success: bool = True
    model_version: str = "nutrisense-scenario-a-v2.0.0"
    feature_schema_version: str = "scenario-a-30-v2"
    predictions: Dict[str, TargetScreeningPrediction]


class HealthResponse(BaseModel):
    """General service health status."""
    status: str = "healthy"


class ModelHealthResponse(BaseModel):
    """Readiness and model integrity diagnostic status."""
    status: str
    model_version: str
    registry_version: str
    feature_schema_version: str
    model_family: str
    targets: List[str]
    model_integrity_verified: bool


class MetadataResponse(BaseModel):
    """Public application and model provenance metadata."""
    application_name: str
    model_version: str
    feature_schema_version: str
    scenario: str
    model_family: str
    target_names: List[str]
    threshold_values: Dict[str, float]
    disclaimer: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[List[Any]] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
