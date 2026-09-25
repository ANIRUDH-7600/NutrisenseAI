# NutriSense AI: Oral Project Explanations & Interview Narratives

Three structured spoken explanations designed for final-year project defense, academic viva examinations, and technical engineering interviews.

---

## 1. The 30-Second Elevator Pitch

> *"NutriSense AI is an explainable machine learning risk intelligence system designed for community pre-screening of early childhood malnutrition in India. In many rural Anganwadi centers, physical weighing scales and height boards are missing or broken. Our system solves this by using 34 non-invasive questions—such as child age, birth history, maternal education, and household sanitation—to predict statistical risk for stunting, underweight, and wasting using LightGBM models trained on over 220,000 children from the NFHS-5 survey. By calibrating decision thresholds specifically for frontline screening, we capture approximately 65% of malnourished children so that health workers can immediately prioritize them for formal physical examination."*

---

## 2. The 2-Minute Executive Summary

> *"Good morning, respected evaluators. My project is NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System.*
> 
> *In India, over one-third of under-five children suffer from stunting or underweight. While routine growth monitoring requires physical weighing scales and infantometers, frontline Anganwadi workers frequently face equipment shortages or broken instruments in remote habitations.
> 
> *To address this, we developed **Scenario A: Scale-Free Community Pre-Screening**. Using the nationally representative NFHS-5 survey covering 220,460 living under-five children, we engineered 34 non-invasive candidate features spanning child demographics, maternal health, household socioeconomics, and recent illness recall. Crucially, all direct physical measurements—like height, weight, and Z-scores—were completely isolated to prevent data leakage.
> 
> *We evaluated five machine learning algorithms using a household-grouped split ensuring zero sibling contamination. **LightGBM Unweighted** proved to be the champion model. Instead of relying on the default 0.50 threshold—which misses most malnourished children—we established validation-derived operating thresholds: 0.35 for stunting, 0.31 for underweight, and 0.17 for wasting. The operating thresholds were selected using the validation cohort in Step 11 and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13 (we do not claim probability calibration).*
> 
> *On the locked, out-of-sample test cohort of over 33,000 children, our models achieved a Sensitivity of 63.42% for stunting (Specificity 61.18%, F1 0.5431, PR-AUC 0.5170, ROC-AUC 0.6671), 64.60% for underweight (Specificity 61.79%, F1 0.5165, PR-AUC 0.4865, ROC-AUC 0.6848), and 69.82% for wasting (Specificity 48.42%, F1 0.3539, PR-AUC 0.2752, ROC-AUC 0.6349). Using TreeSHAP, we verified that child age, birth weight, maternal BMI, and household wealth drive predictions in alignment with established pediatric evidence.
> 
> *Finally, we operationalized the research into a production-grade FastAPI microservice with cryptographic model auditing and a responsive React frontend dashboard. The system does not provide medical diagnoses; rather, it serves as a decision-support filter that flags vulnerable children for confirmatory clinical anthropometry. Thank you."*

---

## 3. The 5-Minute Comprehensive Technical Deep Dive

> *"Respected examiners, I am presenting NutriSense AI: An Explainable Machine Learning System for Non-Invasive Community Pre-Screening of Childhood Undernutrition in India.
> 
> ### 1. Background & Problem Framing
> Anthropometric growth failure among under-five children encompasses three distinct conditions according to WHO 2006 Standards:
> - **Stunting**: Height-for-Age Z-score under -2 SD, reflecting chronic linear growth deficits ($35.5\%$ prevalence in our cohort).
> - **Underweight**: Weight-for-Age Z-score under -2 SD, reflecting composite mass deficits ($30.8\%$ prevalence).
> - **Wasting**: Weight-for-Height Z-score under -2 SD, reflecting acute tissue depletion ($18.7\%$ prevalence).
> 
> Frontline Anganwadi workers (AWWs) are tasked with monthly growth monitoring, but equipment shortages, transport challenges, and calibration drift often leave children unmeasured. Our objective was to investigate whether non-invasive, interview-accessible survey indicators alone could predict risk with sufficient sensitivity to act as an effective community pre-screening triage filter.
> 
> ### 2. Data Engineering & Leakage Isolation
> We ingested the India NFHS-5 (2019–21) Children's Recode dataset containing 232,920 raw records. After excluding deceased children, our analytical cohort comprised 220,460 records.
> 
> We engineered exactly 34 non-invasive features across 8 domains: child demographics, subjective birth size, breastfeeding practices, 2-week symptom recall (diarrhea, fever, cough), maternal profile (age, parity, education, BMI), household wealth quintile, WASH environment (JMP drinking water and sanitation types), and geographic state. 
> 
> To ensure strict scientific validity, we permanently expunged all direct physical anthropometric measurements (`hw2`, `hw3`, `hw70`–`hw73`, `hw57`, `hw13`) and survey sampling weights.
> 
> ### 3. Partitioning Strategy
> Standard random train/test splitting causes sibling contamination because siblings share the same household environment, wealth, and maternal genetics. We solved this by implementing household-clustered splitting on cluster and household IDs `(v001, v002)`, allocating 70% to training ($N = 154{,}238$), 15% to validation ($N = 33{,}153$), and 15% to a locked test partition ($N = 33{,}069$) with **zero shared households**.
> 
> ### 4. Model Benchmarking & Selection
> We benchmarked regularized Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM across both unweighted and class-weighted training. LightGBM Unweighted achieved the best rank discrimination (validation ROC-AUC: 0.6699 for stunting, 0.6812 for underweight, 0.6253 for wasting) and lower Brier scores compared to weighted models, which distorted probability distributions.
> 
> ### 5. Validation-Derived Operating Thresholds
> In community screening, missing a malnourished child (false negative) is clinically unacceptable, whereas a false positive merely costs 5–10 minutes of frontline measurement time. Therefore, validation-derived operating thresholds were selected using the validation cohort in Step 11 via F1 and Youden's $J$ optimization, and the resulting configurations were locked in Step 12 before final evaluation on the held-out test cohort in Step 13: $\tau = 0.35$ for stunting, $\tau = 0.31$ for underweight, and $\tau = 0.17$ for acute wasting (we do not claim probability calibration).
> 
> ### 6. Out-of-Sample Test Generalization
> When evaluated on the sealed test partition ($N_{test} = 33{,}069$):
> - **Stunting** ($\tau = 0.35$): ROC-AUC 0.6671, PR-AUC 0.5170, Sensitivity 63.42% [95% CI: 62.46%–64.31%], Specificity 61.18%, F1 0.5431.
> - **Underweight** ($\tau = 0.31$): ROC-AUC 0.6848, PR-AUC 0.4865, Sensitivity 64.60% [95% CI: 63.56%–65.57%], Specificity 61.79%, F1 0.5165.
> - **Wasting** ($\tau = 0.17$): ROC-AUC 0.6349, PR-AUC 0.2752, Sensitivity 69.82% [95% CI: 68.64%–70.97%], Specificity 48.42%, F1 0.3539.
> Validation-to-test performance shifts were negligible ($|\Delta\text{ROC-AUC}| < 0.0096$), confirming zero test leakage.
> 
> ### 7. Explainability & Subgroup Audits
> Using TreeSHAP on 5,000 validation samples, we audited feature attributions. Child age was the top predictor for stunting, showing cumulative positive risk contributions over time. In contrast, child age exhibited a negative correlation with wasting risk ($\rho = -0.921$), matching clinical reality where acute wasting peaks in early infancy. Maternal BMI and wealth quintile exerted protective negative contributions across all three targets.
> 
> In subgroup analysis, sensitivity was highest in poorest households ($74.5\%$) and rural areas, while affluent households exhibited higher false-negative rates because malnutrition in wealthy homes is driven by unmeasured idiosyncratic factors.
> 
> ### 8. Full-Stack Software Engineering
> To prove deployability, we encapsulated the approved pipelines within a model registry auditing SHA-256 hashes. We built an asynchronous FastAPI backend that verifies model integrity at startup, validates requests via Pydantic (`extra = "forbid"`), logs zero sensitive child data, and serves inference in under 10 ms. We connected this to a responsive, accessible React Single-Page Application featuring a 7-section form, a tri-target results dashboard, and secondary clinical triage protocols.
> 
> ### 9. Limitations & Boundaries
> NutriSense AI is an observational survey screening tool, not a clinical diagnostic replacement. It does not output therapeutic feeding prescriptions or drug dosages. Its value lies in screening triage: prioritizing scarce measuring equipment for children who need it most. Thank you."*
