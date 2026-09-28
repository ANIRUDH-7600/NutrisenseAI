"""Automated Data Dictionary Generator for India NFHS-5 Children's Recode (KR).

Extracts variable metadata directly from IAKR7EFL.DTA and IAKR7EFL.DO to generate
an auditable data_dictionary.csv and detailed markdown documentation.
"""
import re
import pandas as pd
from pandas.io.stata import StataReader

DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
DO_PATH = "IAKR7EDT/IAKR7EFL.DO"
OUTPUT_CSV = "data_dictionary.csv"

# Comprehensive list of candidate variables to document across all required domains
CANDIDATE_VARS_SPEC = [
    # --- CHILD ANTHROPOMETRIC TARGETS & MEASUREMENTS ---
    {
        "name": "hw70", "domain": "Child Anthropometry",
        "role": "candidate target", "unit": "Standard Deviations (SD)",
        "scale": "2 implied decimals (divide by 100)",
        "special_codes": "9996=Height out of limits, 9997=Age out of limits, 9998=WHO Flagged, NaN=Missing",
        "explanation": "Height-for-age standard deviation calculated against the WHO 2006 Child Growth Standards. Used to classify stunting.",
        "ml_usable": "Yes (Target only)",
        "decision_reason": "Primary candidate target for stunting prediction. HAZ < -2.00 SD indicates moderate stunting; < -3.00 SD indicates severe stunting."
    },
    {
        "name": "hw71", "domain": "Child Anthropometry",
        "role": "candidate target", "unit": "Standard Deviations (SD)",
        "scale": "2 implied decimals (divide by 100)",
        "special_codes": "9996=Height out of limits, 9997=Age out of limits, 9998=WHO Flagged, NaN=Missing",
        "explanation": "Weight-for-age standard deviation calculated against the WHO 2006 Child Growth Standards. Used to classify underweight.",
        "ml_usable": "Yes (Target only)",
        "decision_reason": "Primary candidate target for underweight prediction. WAZ < -2.00 SD indicates moderate underweight; < -3.00 SD indicates severe underweight."
    },
    {
        "name": "hw72", "domain": "Child Anthropometry",
        "role": "candidate target", "unit": "Standard Deviations (SD)",
        "scale": "2 implied decimals (divide by 100)",
        "special_codes": "9996=Height out of limits, 9997=Age out of limits, 9998=WHO Flagged, NaN=Missing",
        "explanation": "Weight-for-height standard deviation calculated against the WHO 2006 Child Growth Standards. Used to classify acute wasting.",
        "ml_usable": "Yes (Target only)",
        "decision_reason": "Primary candidate target for acute wasting prediction. WHZ < -2.00 SD indicates moderate wasting; < -3.00 SD indicates severe acute wasting."
    },
    {
        "name": "hw73", "domain": "Child Anthropometry",
        "role": "candidate target", "unit": "Standard Deviations (SD)",
        "scale": "2 implied decimals (divide by 100)",
        "special_codes": "9996=Height out of limits, 9997=Age out of limits, 9998=WHO Flagged, NaN=Missing",
        "explanation": "Body Mass Index (BMI) standard deviation calculated against the WHO 2006 Child Growth Standards.",
        "ml_usable": "Yes (Alternative Target)",
        "decision_reason": "Evaluated as alternative or secondary indicator for acute malnutrition and severe thinness."
    },
    {
        "name": "hw2", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Kilograms (kg)",
        "scale": "1 implied decimal (divide by 10)",
        "special_codes": "9994=Present but not measured, 9995=Refused, 9996=Other, NaN=Missing",
        "explanation": "Direct child weight in kilograms measured using SECA digital scale.",
        "ml_usable": "No (Leakage for WHZ/WAZ)",
        "decision_reason": "Direct input to WAZ and WHZ calculations; strictly excluded from feature set in non-clinical pre-screening models to prevent target leakage."
    },
    {
        "name": "hw3", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Centimeters (cm)",
        "scale": "1 implied decimal (divide by 10)",
        "special_codes": "9994=Present but not measured, 9995=Refused, 9996=Other, NaN=Missing",
        "explanation": "Direct child height/length measured using portable measuring board (recumbent length <24 mo, standing height >=24 mo).",
        "ml_usable": "No (Leakage for HAZ/WHZ)",
        "decision_reason": "Direct mathematical input to HAZ and WHZ calculations; strictly excluded to prevent trivial target leakage."
    },
    {
        "name": "hw4", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentile",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Height/Age percentile according to WHO standard.",
        "ml_usable": "No (Target derivation)",
        "decision_reason": "Mathematical derivative of hw70; prohibited from feature matrix."
    },
    {
        "name": "hw5", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Standard Deviations",
        "scale": "2 implied decimals", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Height/Age standard deviation (NCHS reference).",
        "ml_usable": "No (Legacy target leakage)",
        "decision_reason": "Legacy NCHS standard deviation; redundant and direct target leakage."
    },
    {
        "name": "hw6", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentage (%)",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Height/Age percent of reference median (NCHS).",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Mathematical derivative of height; prohibited."
    },
    {
        "name": "hw7", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentile",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Age percentile according to WHO standard.",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Mathematical derivative of hw71; prohibited."
    },
    {
        "name": "hw8", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Standard Deviations",
        "scale": "2 implied decimals", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Age standard deviation (NCHS reference).",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Legacy NCHS weight-for-age SD; direct target leakage."
    },
    {
        "name": "hw9", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentage (%)",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Age percent of reference median (NCHS).",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Mathematical derivative of weight; prohibited."
    },
    {
        "name": "hw10", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentile",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Height percentile according to WHO standard.",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Mathematical derivative of hw72; prohibited."
    },
    {
        "name": "hw11", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Standard Deviations",
        "scale": "2 implied decimals", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Height standard deviation (NCHS reference).",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Legacy NCHS weight-for-height SD; direct target leakage."
    },
    {
        "name": "hw12", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Percentage (%)",
        "scale": "None", "special_codes": "9998=Flagged, NaN=Missing",
        "explanation": "Weight/Height percent of reference median (NCHS).",
        "ml_usable": "No (Leakage)",
        "decision_reason": "Mathematical derivative of weight and height; prohibited."
    },
    {
        "name": "hw13", "domain": "Child Anthropometry",
        "role": "potential leakage variable", "unit": "Categorical code",
        "scale": "None", "special_codes": "1=Dead, 2=Sick, 3=Not present, 4=Refused, 5=Mother refused, 6=Other, 7=No measurement",
        "explanation": "Audit result of anthropometric measurement attempt.",
        "ml_usable": "No (Data audit only)",
        "decision_reason": "Encodes whether target was measured or refused; used in Step 3 population filtering, excluded from features."
    },

    # --- CHILD DEMOGRAPHICS & MORBIDITY ---
    {
        "name": "hw1", "domain": "Child Demographics",
        "role": "candidate feature", "unit": "Months",
        "scale": "None", "special_codes": "NaN=Missing/Not measured",
        "explanation": "Child's exact age in completed months at the time of the survey interview.",
        "ml_usable": "Yes",
        "decision_reason": "Core predictor. Undernutrition patterns (stunting vs wasting) vary dramatically by developmental stage (0-5, 6-23, 24-59 months)."
    },
    {
        "name": "b4", "domain": "Child Demographics",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "None (1=Male, 2=Female)",
        "explanation": "Biological sex of the child.",
        "ml_usable": "Yes",
        "decision_reason": "Universal demographic predictor; epidemiological studies reveal consistent sex-specific disparities in malnutrition risk."
    },
    {
        "name": "b5", "domain": "Child Demographics",
        "role": "survey-design variable", "unit": "Binary code",
        "scale": "None", "special_codes": "0=No, 1=Yes",
        "explanation": "Child alive status at time of survey.",
        "ml_usable": "No (Eligibility filter)",
        "decision_reason": "Mandatory population eligibility filter. Children who are deceased cannot be screened for current nutritional deficit."
    },
    {
        "name": "b8", "domain": "Child Demographics",
        "role": "potentially derived variable", "unit": "Years",
        "scale": "None", "special_codes": "NaN=Deceased child",
        "explanation": "Current age of child in completed single years.",
        "ml_usable": "No (Coarse granularity)",
        "decision_reason": "Coarser version of hw1 (age in months); hw1 provides superior resolution."
    },
    {
        "name": "bord", "domain": "Child Demographics",
        "role": "candidate feature", "unit": "Count",
        "scale": "None", "special_codes": "None",
        "explanation": "Birth order number of the child relative to mother's full birth history.",
        "ml_usable": "Yes",
        "decision_reason": "Direct predictor; higher birth order is linked with intra-household resource dilution and increased malnutrition risk."
    },
    {
        "name": "b0", "domain": "Child Demographics",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "0=Single, 1=Twin/Multiple",
        "explanation": "Indicator if the child was born as part of a multiple birth (twin, triplet).",
        "ml_usable": "Yes",
        "decision_reason": "Twin births have significantly elevated risk of low birth weight, pre-term delivery, and growth faltering."
    },
    {
        "name": "b11", "domain": "Child Demographics",
        "role": "candidate feature", "unit": "Months",
        "scale": "None", "special_codes": "NaN=Firstborn child",
        "explanation": "Preceding birth interval in completed months.",
        "ml_usable": "Yes",
        "decision_reason": "Short birth spacing (<24 months) depletes maternal nutritional reserves, worsening fetal and infant growth."
    },
    {
        "name": "m18", "domain": "Child Birth History",
        "role": "candidate feature", "unit": "Ordinal scale",
        "scale": "None", "special_codes": "8=Don't know, 9=Missing",
        "explanation": "Subjective size of child at birth reported by mother (1=Very large to 5=Very small).",
        "ml_usable": "Yes",
        "decision_reason": "Crucial proxy for intrauterine growth restriction, especially in rural settings where digital birth scales are unavailable."
    },
    {
        "name": "m19", "domain": "Child Birth History",
        "role": "candidate feature", "unit": "Kilograms",
        "scale": "3 implied decimals (divide by 1000)",
        "special_codes": "9996=Not weighed, 9998=Don't know",
        "explanation": "Recorded birth weight from child immunization card or mother recall.",
        "ml_usable": "Yes (Subject to missingness handling)",
        "decision_reason": "Direct clinical indicator of Low Birth Weight (<2500g), the strongest epidemiological predictor of stunting."
    },
    {
        "name": "hw57", "domain": "Child Health",
        "role": "candidate feature", "unit": "Ordinal code",
        "scale": "None", "special_codes": "NaN=Not measured",
        "explanation": "Anemia level of child (1=Severe, 2=Moderate, 3=Mild, 4=Not anemic).",
        "ml_usable": "Yes (Candidate health predictor)",
        "decision_reason": "Captures micronutrient deficiency (iron deficiency anemia) frequently comorbid with protein-energy malnutrition."
    },
    {
        "name": "v404", "domain": "Child Feeding",
        "role": "candidate feature", "unit": "Binary code",
        "scale": "None", "special_codes": "None (0=No, 1=Yes)",
        "explanation": "Child is currently being breastfed.",
        "ml_usable": "Yes",
        "decision_reason": "Protective factor against acute infections and wasting during early infancy."
    },
    {
        "name": "m4", "domain": "Child Feeding",
        "role": "candidate feature", "unit": "Months",
        "scale": "None", "special_codes": "93=Ever breastfed not current, 94=Never, 95=Still breastfeeding, 98=Don't know",
        "explanation": "Total duration of breastfeeding in months.",
        "ml_usable": "Yes",
        "decision_reason": "Captures adherence to WHO infant and young child feeding guidelines."
    },
    {
        "name": "h11", "domain": "Child Morbidity",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "0=No, 2=Yes, 8=Don't know, NaN=Deceased",
        "explanation": "Child experienced acute diarrhea in the two weeks preceding the survey.",
        "ml_usable": "Yes",
        "decision_reason": "Severe acute enteric infection directly causes fluid loss, malabsorption, and acute wasting."
    },
    {
        "name": "h22", "domain": "Child Morbidity",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "0=No, 2=Yes, 8=Don't know, NaN=Deceased",
        "explanation": "Child experienced fever in the two weeks preceding the survey.",
        "ml_usable": "Yes",
        "decision_reason": "Systemic inflammatory/infectious marker that suppresses appetite and accelerates metabolic catabolism."
    },
    {
        "name": "h31", "domain": "Child Morbidity",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "0=No, 2=Yes, 8=Don't know, NaN=Deceased",
        "explanation": "Child experienced acute cough/respiratory illness in the two weeks preceding the survey.",
        "ml_usable": "Yes",
        "decision_reason": "Acute respiratory infections impair nutritional recovery and reflect household environmental vulnerability."
    },

    # --- MOTHER CHARACTERISTICS ---
    {
        "name": "v012", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Years",
        "scale": "None", "special_codes": "None (15 to 49)",
        "explanation": "Mother's current age in completed years at interview.",
        "ml_usable": "Yes",
        "decision_reason": "Maternal physiological maturity directly correlates with maternal nutrition and care practices."
    },
    {
        "name": "v212", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Years",
        "scale": "None", "special_codes": "None",
        "explanation": "Mother's age at first birth.",
        "ml_usable": "Yes",
        "decision_reason": "Adolescent pregnancy (<19 years) strongly associates with intergenerational transmission of undernutrition."
    },
    {
        "name": "v445", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "kg/m^2",
        "scale": "2 implied decimals (divide by 100)",
        "special_codes": "9998=Flagged, NaN=Not measured",
        "explanation": "Mother's Body Mass Index (BMI).",
        "ml_usable": "Yes",
        "decision_reason": "Key biological predictor; maternal undernutrition (BMI < 18.5) strongly predicts child stunting and wasting."
    },
    {
        "name": "v457", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Ordinal code",
        "scale": "None", "special_codes": "NaN=Not measured",
        "explanation": "Mother's anemia level (1=Severe, 2=Moderate, 3=Mild, 4=Not anemic).",
        "ml_usable": "Yes",
        "decision_reason": "Indicates maternal chronic micronutrient deficiency and household diet quality."
    },
    {
        "name": "v106", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Ordinal code",
        "scale": "None", "special_codes": "None (0=None, 1=Primary, 2=Secondary, 3=Higher)",
        "explanation": "Highest educational level attained by mother.",
        "ml_usable": "Yes",
        "decision_reason": "Major socio-behavioral predictor; maternal literacy governs health-seeking behaviors and optimal child feeding."
    },
    {
        "name": "v701", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Ordinal code",
        "scale": "None", "special_codes": "8=Don't know, 9=Missing",
        "explanation": "Husband/partner's highest educational attainment.",
        "ml_usable": "Yes",
        "decision_reason": "Paternal education determines household economic capacity and health literacy."
    },
    {
        "name": "v714", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Binary code",
        "scale": "None", "special_codes": "None (0=No, 1=Yes)",
        "explanation": "Mother is currently working / employed.",
        "ml_usable": "Yes",
        "decision_reason": "Reflects household dual income and maternal time allocation for child care."
    },
    {
        "name": "m14", "domain": "Maternal Healthcare",
        "role": "candidate feature", "unit": "Visits count",
        "scale": "None", "special_codes": "98=Don't know, NaN=Missing",
        "explanation": "Number of antenatal care (ANC) visits during pregnancy.",
        "ml_usable": "Yes",
        "decision_reason": "Direct proxy for institutional health access, maternal supplementation, and prenatal monitoring."
    },
    {
        "name": "m15", "domain": "Maternal Healthcare",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "96=Other",
        "explanation": "Place of delivery (11=Home, 20+=Public sector, 30+=Private sector).",
        "ml_usable": "Yes",
        "decision_reason": "Institutional delivery indicates access to sanitary birth environments and neonatal care."
    },
    {
        "name": "v201", "domain": "Maternal Characteristics",
        "role": "candidate feature", "unit": "Count",
        "scale": "None", "special_codes": "None",
        "explanation": "Total number of children ever born to mother.",
        "ml_usable": "Yes",
        "decision_reason": "Parity and fertility pressure indicator influencing household dependency ratio."
    },

    # --- HOUSEHOLD / SOCIOECONOMIC & SANITATION ---
    {
        "name": "v190", "domain": "Household Socioeconomic",
        "role": "candidate feature", "unit": "Quintile (1-5)",
        "scale": "None", "special_codes": "None (1=Poorest to 5=Richest)",
        "explanation": "Household wealth index quintile constructed from asset ownership.",
        "ml_usable": "Yes",
        "decision_reason": "Fundamental social determinant of health, food purchasing power, and living standard."
    },
    {
        "name": "v025", "domain": "Household Socioeconomic",
        "role": "candidate feature", "unit": "Binary code",
        "scale": "None", "special_codes": "None (1=Urban, 2=Rural)",
        "explanation": "Type of place of residence.",
        "ml_usable": "Yes",
        "decision_reason": "Captures urban-rural infrastructure, sanitation, and health facility accessibility disparities."
    },
    {
        "name": "v024", "domain": "Geographic / Regional",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "None (1 to 37 Indian States/UTs)",
        "explanation": "State or Union Territory of residence in India.",
        "ml_usable": "Yes",
        "decision_reason": "Crucial for capturing regional epidemiological gradients across India (e.g., northern vs southern states)."
    },
    {
        "name": "sdist", "domain": "Geographic / Regional",
        "role": "survey-design variable", "unit": "Categorical code",
        "scale": "None", "special_codes": "None",
        "explanation": "District identification code within state.",
        "ml_usable": "No (High cardinality identifier)",
        "decision_reason": "High-cardinality district identifier (>700 categories); better suited for spatial aggregation than raw tabular features."
    },
    {
        "name": "s116", "domain": "Social & Cultural",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "8=Don't know",
        "explanation": "Belonging to a Scheduled Caste (SC), Scheduled Tribe (ST), Other Backward Class (OBC), or None.",
        "ml_usable": "Yes",
        "decision_reason": "India-specific structural determinant reflecting systemic historical inequities in nutrition and healthcare access."
    },
    {
        "name": "v130", "domain": "Social & Cultural",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "96=Other",
        "explanation": "Religion of household respondent (1=Hindu, 2=Muslim, 3=Christian, 4=Sikh, etc.).",
        "ml_usable": "Yes",
        "decision_reason": "Sociocultural customs, fasting patterns, and dietary habits vary across religious communities."
    },
    {
        "name": "v113", "domain": "Water & Sanitation",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "96=Other, 97=Not a de jure resident",
        "explanation": "Main source of drinking water for household members.",
        "ml_usable": "Yes",
        "decision_reason": "Unimproved water sources directly drive waterborne diarrheal pathogens and environmental enteric dysfunction."
    },
    {
        "name": "v116", "domain": "Water & Sanitation",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "96=Other, 97=Not a de jure resident",
        "explanation": "Type of toilet facility utilized by household.",
        "ml_usable": "Yes",
        "decision_reason": "Open defecation and unimproved sanitation cause chronic intestinal inflammation, severely impeding nutrient absorption."
    },
    {
        "name": "v119", "domain": "Household Environment",
        "role": "candidate feature", "unit": "Binary code",
        "scale": "None", "special_codes": "7=Not a de jure resident",
        "explanation": "Household has electricity connection.",
        "ml_usable": "Yes",
        "decision_reason": "Proxy for modern infrastructure, refrigeration capability, and household standard of living."
    },
    {
        "name": "v161", "domain": "Household Environment",
        "role": "candidate feature", "unit": "Categorical code",
        "scale": "None", "special_codes": "96=Other",
        "explanation": "Type of cooking fuel utilized (biomass/wood vs LPG/electricity).",
        "ml_usable": "Yes",
        "decision_reason": "Indoor air pollution from biomass fuel increases child respiratory infection risk, compounding nutritional failure."
    },
    {
        "name": "v136", "domain": "Household Demographics",
        "role": "candidate feature", "unit": "Count",
        "scale": "None", "special_codes": "None",
        "explanation": "Number of usual household members listed in household schedule.",
        "ml_usable": "Yes",
        "decision_reason": "Household crowding and resource competition measure."
    },
    {
        "name": "v151", "domain": "Household Demographics",
        "role": "candidate feature", "unit": "Binary code",
        "scale": "None", "special_codes": "None (1=Male, 2=Female)",
        "explanation": "Sex of head of household.",
        "ml_usable": "Yes",
        "decision_reason": "Female-headed households frequently reflect distinct expenditure allocations towards child nutrition and health."
    },

    # --- SURVEY DESIGN & IDENTIFIERS ---
    {
        "name": "v001", "domain": "Survey Design",
        "role": "identifier", "unit": "ID number",
        "scale": "None", "special_codes": "None",
        "explanation": "Cluster number / Primary Sampling Unit (PSU).",
        "ml_usable": "No (Survey identifier)",
        "decision_reason": "Sampling unit identifier; used for survey clustering and spatial aggregation, never as a direct predictive feature."
    },
    {
        "name": "v002", "domain": "Survey Design",
        "role": "identifier", "unit": "ID number",
        "scale": "None", "special_codes": "None",
        "explanation": "Household number within cluster.",
        "ml_usable": "No (Identifier)",
        "decision_reason": "Arbitrary survey administration ID."
    },
    {
        "name": "v003", "domain": "Survey Design",
        "role": "identifier", "unit": "ID number",
        "scale": "None", "special_codes": "None",
        "explanation": "Respondent's line number in household schedule.",
        "ml_usable": "No (Identifier)",
        "decision_reason": "Arbitrary administrative line number."
    },
    {
        "name": "v005", "domain": "Survey Design",
        "role": "survey weight", "unit": "Probability Weight",
        "scale": "6 implied decimals (divide by 1,000,000)",
        "special_codes": "None",
        "explanation": "Women's individual sample weight for adjusting non-response and unequal selection probabilities.",
        "ml_usable": "No (Feature), Yes (Weighted Evaluation)",
        "decision_reason": "Survey sampling weight; must not be fed as an input feature to prevent model learning sampling artifacts."
    },
    {
        "name": "v021", "domain": "Survey Design",
        "role": "survey-design variable", "unit": "PSU code",
        "scale": "None", "special_codes": "None",
        "explanation": "Primary sampling unit (PSU) code.",
        "ml_usable": "No",
        "decision_reason": "Survey complex variance estimation parameter."
    },
    {
        "name": "v022", "domain": "Survey Design",
        "role": "survey-design variable", "unit": "Stratum code",
        "scale": "None", "special_codes": "None",
        "explanation": "Sample strata for sampling errors.",
        "ml_usable": "No",
        "decision_reason": "Sampling strata identifier used in complex survey statistical analysis."
    }
]

def generate_dictionary():
    print(f"Loading metadata from {DTA_PATH} and {DO_PATH}...")
    with StataReader(DTA_PATH) as reader:
        vlabels = reader.variable_labels()
        val_labels = reader.value_labels()

    # Read .DO file for full value label text
    with open(DO_PATH, "r", encoding="latin1") as f:
        do_content = f.read()

    # Load the specific columns to get empirical counts across all 232,920 records
    var_names = [item["name"] for item in CANDIDATE_VARS_SPEC]
    print(f"Reading empirical values for {len(var_names)} candidate variables...")
    df = pd.read_stata(DTA_PATH, columns=var_names, convert_categoricals=False)

    rows_out = []
    for item in CANDIDATE_VARS_SPEC:
        var = item["name"]
        series = df[var]
        dhs_label = vlabels.get(var, "No label in metadata")
        
        # Calculate valid vs missing/special counts
        n_nan = int(series.isna().sum())
        n_total = len(series)
        n_valid = n_total - n_nan
        
        # Parse value labels from DO file if categorical
        val_lbl_match = re.search(rf'label values {var}\s+([A-Za-z0-9_]+)', do_content, re.IGNORECASE)
        val_codes_str = "Continuous / Numerical"
        if val_lbl_match:
            lbl_name = val_lbl_match.group(1)
            def_match = re.search(rf'label define {lbl_name}\s+([^;\n]+(?:\n\s+[^;\n]+)*)', do_content, re.IGNORECASE)
            if def_match:
                pairs = re.findall(r'(\d+)\s+"([^"]+)"', def_match.group(1))
                if len(pairs) <= 6:
                    val_codes_str = "; ".join([f"{c}={txt}" for c, txt in pairs])
                else:
                    val_codes_str = "; ".join([f"{c}={txt}" for c, txt in pairs[:5]]) + f"; ... ({len(pairs)} categories)"

        rows_out.append({
            "variable_name": var,
            "official_dhs_label": dhs_label,
            "domain": item["domain"],
            "data_type": str(series.dtype),
            "unit": item["unit"],
            "implied_decimal_scaling": item["scale"],
            "value_categorical_codes": val_codes_str,
            "missing_special_codes": item["special_codes"],
            "valid_observations_count": n_valid,
            "missing_observations_count": n_nan,
            "role": item["role"],
            "explanation": item["explanation"],
            "ml_usable": item["ml_usable"],
            "decision_reason": item["decision_reason"]
        })

    dict_df = pd.DataFrame(rows_out)
    dict_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] data_dictionary.csv successfully generated with {len(dict_df)} documented variables!")
    return dict_df

if __name__ == "__main__":
    generate_dictionary()
