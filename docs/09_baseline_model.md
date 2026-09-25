# Step 9 — Baseline Model Documentation

**NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System**  
*Scenario A — Community Pre-Screening Setting*  
*Dataset: NFHS-5 India (2019–21) Children's Recode (KR)*

---

## 1. Step 9 Objective

The primary objective of Step 9 is to build, validate, and document a simple, interpretable, and reproducible **Baseline Model** for predicting childhood malnutrition risk in India. By establishing an empirical performance benchmark using **Logistic Regression** on the approved Scenario A feature set, Step 9 provides a transparent reference point against which complex non-linear algorithms (e.g., LightGBM, XGBoost, Random Forests, Multi-Layer Perceptrons) will be compared in subsequent steps.

---

## 2. Research Question Addressed

> *To what extent can demographic, maternal, birth history, and household environmental features—obtained strictly without physical anthropometric measurements—predict childhood stunting, underweight, and wasting under a linear log-odds formulation in a community pre-screening context?*

---

## 3. Why Logistic Regression Was Selected as the Baseline

1. **Standard Scientific Reference**: In epidemiology, biostatistics, and clinical risk intelligence, multivariable logistic regression is the gold-standard linear parametric baseline for binary disease outcomes.
2. **High Interpretability**: Coefficients correspond directly to log-odds ratios ($\beta_j = \ln(\text{OR}_j)$), providing a mathematically grounded sanity check on whether predictor directions align with established pediatric literature (e.g., higher wealth or maternal education reducing risk).
3. **No Non-Linear Assumptions**: It models the log-odds as an additive linear combination, establishing the performance achievable *without* tree interactions or non-linear manifolds.
4. **Computational Efficiency & Determinism**: Fast, convex optimization guarantees global convergence without stochastic training instability.

---

## 4. Why a Baseline is Necessary Before Advanced Models

In applied machine learning for healthcare, deploying complex gradient-boosted trees or deep neural networks without a baseline is bad scientific practice. A baseline is essential because:
1. **Quantifies Added Value**: It determines whether non-linear algorithms provide genuine performance gains or merely add unneeded complexity.
2. **Exposes Linear Separability**: If a simple logistic regression achieves an AUC of 0.67, we immediately understand the baseline discriminative signal inherent in the survey predictors.
3. **Validates the End-to-End Pipeline**: It verifies data loading, feature transformation, leakage isolation, cross-validation, and metric computation before introducing complex hyperparameter search spaces.

---

## 5. Dataset Used

- **Data Source**: Demographic and Health Surveys (DHS) / National Family Health Survey (NFHS-5, 2019–21) India Children's Recode (`IAKR7EFL.DTA`).
- **Research Population**: Living children aged 0–59 completed months ($N = 221{,}263$).
- **Partitioning**: Step 8 household-grouped split (70% Train, 15% Validation, 15% Test).

---

## 6. Scenario A Prediction Setting

- **Intended Use Case**: Community-level pre-screening by frontline health workers (ASHAs / Anganwadi workers) in rural or low-resource settings.
- **Physical Equipment Constraint**: Zero reliance on infant scales (weight) or stadiometers/infantometers (height/length).
- **Prohibited Variables**: All physical measurements (`hw2`, `hw3`, `hw4`–`hw12`) and continuous anthropometric z-scores (`hw70`–`hw73`) are strictly barred from the feature matrix $X$.

---

## 7. 34 Approved Features

The model utilizes exclusively the 34 approved candidate features from Step 7:
1. **Child Demographics (7)**: `child_age_months`, `child_age_group`, `child_sex_male`, `birth_order`, `is_multiple_birth`, `is_firstborn`, `preceding_birth_interval_months`
2. **Birth Characteristics (4)**: `birth_size_ordinal`, `birth_weight_kg`, `birth_weight_missing`, `delivery_place_type`
3. **Breastfeeding / Feeding (1)**: `still_breastfeeding`
4. **Child Health / Morbidity (3)**: `diarrhea_recent`, `fever_recent`, `cough_recent`
5. **Maternal Characteristics (8)**: `mother_age_years`, `mother_age_first_birth`, `mother_education_level`, `mother_bmi`, `mother_bmi_missing`, `total_children_born`, `anc_visits_count`, `anc_visits_missing`
6. **Household Socioeconomic (4)**: `wealth_quintile`, `is_rural`, `caste_category`, `religion_category`
7. **Household Environment & WASH (6)**: `drinking_water_type`, `sanitation_facility_type`, `has_electricity`, `clean_cooking_fuel`, `household_size`, `household_head_female`
8. **Geographic Characteristics (1)**: `state_id`

---

## 8. Target Definitions

NutriSense AI models three distinct clinical undernutrition outcomes based on WHO 2006 Child Growth Standards ($z < -2.0$ SD):
- **Stunting** ($y_S \in \{0, 1\}$): Height-for-Age Z-score ($\text{HAZ} < -2.00$). Chronic malnutrition / linear growth failure.
- **Underweight** ($y_U \in \{0, 1\}$): Weight-for-Age Z-score ($\text{WAZ} < -2.00$). Composite chronic and/or acute deficit.
- **Wasting** ($y_W \in \{0, 1\}$): Weight-for-Height Z-score ($\text{WHZ} < -2.00$). Acute nutritional deficit / acute tissue wasting.

---

## 9. Target-Specific Cohorts

Due to selective anthropometric measurement validity in NFHS-5, each target maintains its own validated cohort:
- **Stunting Model**:
  - Training cohort: $N = 144{,}368$ (Positives: 51,148 | Negatives: 93,220 | Prevalence: 35.43%)
  - Validation cohort: $N = 30{,}885$ (Positives: 10,963 | Negatives: 19,922 | Prevalence: 35.50%)
- **Underweight Model**:
  - Training cohort: $N = 147{,}560$ (Positives: 45,664 | Negatives: 101,896 | Prevalence: 30.95%)
  - Validation cohort: $N = 31{,}531$ (Positives: 9,673 | Negatives: 21,858 | Prevalence: 30.68%)
- **Wasting Model**:
  - Training cohort: $N = 141{,}328$ (Positives: 26,291 | Negatives: 115,037 | Prevalence: 18.60%)
  - Validation cohort: $N = 30{,}211$ (Positives: 5,635 | Negatives: 24,576 | Prevalence: 18.65%)

---

## 10. Train / Validation / Test Strategy

The project employs a household-grouped 70% / 15% / 15% partition scheme established in Step 8:
- **Training (70%)**: Used solely to fit preprocessing transformers and model coefficients.
- **Validation (15%)**: Used solely to evaluate model performance, assess generalization, and compare algorithms.
- **Test (15%)**: Sequestered and locked.

---

## 11. Why the Test Set Was Not Used

> [!IMPORTANT]
> **Test-Set Lock Declaration**:  
> *"The test set was not used for baseline model fitting, preprocessing, model selection, threshold selection, or performance evaluation."*

Evaluating on the test set during model building leads to subtle human snooping and iterative bias. The test set ($N = 33{,}069$) will be evaluated exactly once at the conclusion of the research project on the final selected model architectures.

---

## 12. Preprocessing Pipeline

All feature transformations are encapsulated inside an `sklearn.pipeline.Pipeline` with a `ColumnTransformer`. This architectural design guarantees that no test or validation statistics leak into the training process:

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
])
```

---

## 13. Missing-Value Handling

Following the approved Step 7 feature strategy:
1. **Informative Missingness**: Features with systematic non-response (`birth_weight_kg`, `mother_bmi`, `anc_visits_count`) are explicitly paired with dedicated binary missing indicators (`birth_weight_missing`, `mother_bmi_missing`, `anc_visits_missing`), and their missing continuous values are imputed with the median of `X_train`.
2. **Firstborn Structural Missingness**: `preceding_birth_interval_months` is missing for firstborns; it is paired with `is_firstborn` and imputed with the training median.
3. **Categorical Non-Response**: Missing categories in social groups or WASH are preserved via explicit categorical bins (`'Missing_Caste'`, `'Open_Defecation_None'`).

---

## 14. Numerical Feature Handling

The 27 numerical, ordinal, and binary features are passed through:
- `SimpleImputer(strategy='median')`: Imputes missing continuous values using the training-set median.
- `StandardScaler()`: Subtracts the training mean ($\mu_{\text{train}}$) and scales to unit variance ($\sigma_{\text{train}}$). This ensures that gradient descent and $L_2$ regularization penalize all features uniformly.

---

## 15. Categorical Feature Handling

Categorical features with discrete levels are passed through:
- `SimpleImputer(strategy='most_frequent')`: Fallback imputer for unseen missing values.
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Generates binary indicator columns for each category level.

---

## 16. One-Hot Encoding

The 7 categorical features are expanded into binary indicator columns:
- `child_age_group` (6 bins)
- `delivery_place_type` (3 bins)
- `caste_category` (5 bins)
- `religion_category` (5 bins)
- `drinking_water_type` (3 bins)
- `sanitation_facility_type` (4 bins)
- `state_id` (36 administrative units)

This expands the input matrix from **34 raw features** to **89 encoded numerical features** after transformation. `handle_unknown='ignore'` ensures that any unobserved category in validation data is safely encoded as all zeros without raising runtime errors.

---

## 17. Logistic Regression Explanation

Logistic Regression models the probability $P(Y=1|X)$ via the standard logistic sigmoid function:
$$P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z$ is the linear predictor:
$$z = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
The log-odds (logit) is strictly linear in the parameters:
$$\text{logit}(P) = \ln\left(\frac{P}{1 - P}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$

---

## 18. What `model.fit()` is Doing

Calling `pipeline.fit(X_train, y_train)`:
1. Calculates numerical medians and categorical modes strictly on `X_train`.
2. Fits the `StandardScaler` ($\mu, \sigma$) and `OneHotEncoder` categories on `X_train`.
3. Transforms `X_train` into the 89-dimensional design matrix.
4. Minimizes the $L_2$-regularized binary cross-entropy loss function via the L-BFGS quasi-Newton solver:
   $$\mathcal{L}(\beta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right] + \frac{1}{2C} \|\beta\|_2^2$$
5. Determines optimal weight vector $\beta^*$ within 1,000 iterations.

---

## 19. What `predict()` is Doing

Calling `pipeline.predict(X_val)`:
1. Applies training-fitted transformers to `X_val`.
2. Computes linear score $z_i$ and probability $p_i = \sigma(z_i)$.
3. Applies the default hard threshold cutoff:
   $$\hat{y}_i = \begin{cases} 1 & \text{if } p_i \ge 0.5 \\ 0 & \text{if } p_i < 0.5 \end{cases}$$

---

## 20. What `predict_proba()` is Doing

Calling `pipeline.predict_proba(X_val)`:
- Computes and returns the continuous estimated risk probabilities $[1 - p_i, p_i]$.
- In NutriSense AI, $p_i$ represents the estimated probability that child $i$ is stunted, underweight, or wasted given their community pre-screening profile.

---

## 21. Classification Threshold = 0.5

The default decision rule in binary classification is $\tau = 0.5$. However, when disease prevalence is below 50% (35.5% for Stunting, 30.7% for Underweight, 18.7% for Wasting), uncalibrated probabilities rarely exceed 0.5. As demonstrated below, evaluating imbalanced clinical targets at $\tau = 0.5$ yields high specificity but depressed sensitivity (Recall). Threshold calibration is addressed in future steps.

---

## 22. Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Measures the overall fraction of correct predictions across both classes.

---

## 23. Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$
The probability that a child flagged as high-risk truly has the malnutrition condition. Reflects positive predictive value (PPV).

---

## 24. Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$
The probability that an undernourished child is successfully identified by the screener. Reflects clinical sensitivity.

---

## 25. F1-Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Harmonic mean balancing precision and recall.

---

## 26. ROC-AUC

Area Under the Receiver Operating Characteristic Curve. Evaluates the model's ability to rank a randomly chosen positive case higher than a randomly chosen negative case across all possible probability thresholds ($0.0 \le \tau \le 1.0$). A random classifier scores 0.50; a perfect classifier scores 1.00. Threshold-independent.

---

## 27. PR-AUC

Area Under the Precision-Recall Curve (Average Precision). Evaluates precision across all recall levels. Crucial for imbalanced medical data, where the baseline random score equals the actual class prevalence.

---

## 28. Confusion Matrix

Structured contingency table:
$$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$
- $TN$: True Negatives (Healthy children correctly classified)
- $FP$: False Positives (Healthy children mistakenly flagged as high-risk)
- $FN$: False Negatives (Undernourished children missed by the model)
- $TP$: True Positives (Undernourished children correctly identified)

---

## 29. Why Accuracy Alone is Insufficient

Accuracy is an inadequate metric for disease screening under class imbalance. In Wasting, where the prevalence is 18.65%:
- A naive dummy model that predicts "0" (healthy) for all children achieves an **Accuracy of 81.35%**, while missing **100% of wasted children** (Recall = 0.00%).
- Relying on accuracy in public health would mask complete clinical failure. Evaluators must prioritize **ROC-AUC**, **PR-AUC**, and **Recall** at calibrated risk thresholds.

---

## 30. Results for Stunting

- **Validation Cohort**: $N = 30{,}885$ (Positives: 10,963 | Negatives: 19,922 | Prevalence: 35.50%)
- **ROC-AUC**: **0.6607**
- **PR-AUC**: **0.5056** (vs random prevalence baseline of 0.3550)
- **Accuracy**: **0.6674**
- **Precision**: **0.5780**
- **Recall**: **0.2334**
- **F1-Score**: **0.3326**
- **Confusion Matrix**:
  $$\begin{bmatrix} 18{,}054 & 1{,}868 \\ 8{,}404 & 2{,}559 \end{bmatrix}$$

---

## 31. Results for Underweight

- **Validation Cohort**: $N = 31{,}531$ (Positives: 9,673 | Negatives: 21,858 | Prevalence: 30.68%)
- **ROC-AUC**: **0.6733**
- **PR-AUC**: **0.4678** (vs random prevalence baseline of 0.3068)
- **Accuracy**: **0.7038**
- **Precision**: **0.5582**
- **Recall**: **0.1650**
- **F1-Score**: **0.2547**
- **Confusion Matrix**:
  $$\begin{bmatrix} 20{,}595 & 1{,}263 \\ 8{,}077 & 1{,}596 \end{bmatrix}$$

---

## 32. Results for Wasting

- **Validation Cohort**: $N = 30{,}211$ (Positives: 5,635 | Negatives: 24,576 | Prevalence: 18.65%)
- **ROC-AUC**: **0.6206**
- **PR-AUC**: **0.2596** (vs random prevalence baseline of 0.1865)
- **Accuracy**: **0.8135**
- **Precision**: **0.5000**
- **Recall**: **0.0007** (4 detected cases at $\tau=0.5$)
- **F1-Score**: **0.0014**
- **Confusion Matrix**:
  $$\begin{bmatrix} 24{,}572 & 4 \\ 5{,}631 & 4 \end{bmatrix}$$

---

## 33. Interpretation of the Baseline Results

1. **Discriminative Capacity**:
   - The linear baseline achieves moderate discriminative ranking ability on unseen validation data: Underweight ROC-AUC = **0.6733**, Stunting ROC-AUC = **0.6607**, Wasting ROC-AUC = **0.6206**.
   - All models exceed random guessing by substantial margins (PR-AUC is 15 percentage points above baseline for Stunting and 16 percentage points above baseline for Underweight).
2. **Chronic vs. Acute Deficit Predictability**:
   - Stunting and Underweight (chronic conditions shaped by cumulative household poverty, maternal BMI, and sanitation) are more predictable from socio-demographic features than Wasting (an acute condition driven by recent infectious illness, seasonal shocks, and acute food shortages).
3. **Threshold Behavior**:
   - At the default threshold ($\tau = 0.5$), the models exhibit high specificity but low recall. For Wasting, only 4 of 5,635 cases are classified positive at $\tau = 0.5$. This demonstrates that a fixed 0.5 cutoff is clinically unviable for low-prevalence screening and motivates downstream threshold tuning.

---

## 34. Limitations

1. **Linearity Assumption**: Logistic Regression cannot automatically capture complex non-linear predictor interactions (e.g., synergistic risk between low birthweight and unimproved water).
2. **Uncalibrated Operating Point**: Standard 0.5 thresholding causes severe under-detection of positives.
3. **Survey Measurement Granularity**: Diet and feeding practices are limited in the KR survey file compared to 24-hour dietary recall records.

---

## 35. Leakage Controls

Step 9 enforced four rigorous leakage controls:
1. **Pipeline Encapsulation**: Imputers, scalers, and encoders were embedded within the `Pipeline` and fitted strictly on `X_train`.
2. **Partition Separation**: `(X_train, y_train)` and `(X_val, y_val)` masks were strictly verified disjoint.
3. **Test-Set Lock**: The Test partition ($N = 33{,}069$) was completely inaccessible during training and evaluation.
4. **Feature Isolation**: Verified that zero target columns, household IDs, cluster IDs, or direct anthropometrics exist in $X$.

---

## 36. Comparison with the Selected Base Paper

### Reference: Islam et al. (2024), PLOS ONE
*"Prediction of undernutrition and identification of its influencing predictors among under-five children in Bangladesh using explainable machine learning algorithms"*

---

## 37. What is Similar to the Base Paper

1. Both studies utilize Demographic and Health Surveys (DHS) microdata (Islam et al.: BDHS 2017–18; NutriSense AI: NFHS-5 2019–21).
2. Both studies predict the three standard WHO malnutrition targets: Stunting, Underweight, and Wasting.
3. Both studies formulate the problem as tabular binary classification and evaluate ROC-AUC, Accuracy, Precision, Recall, and F1.

---

## 38. What is Different in NutriSense AI

1. **Scale & Geography**: NutriSense AI analyzes **221,263 Indian under-five children** across 36 states/UTs, compared to 8,759 children in Islam et al.
2. **Explicit Community Pre-Screening Setting**: NutriSense AI strictly adheres to **Scenario A**, banning physical scales and stadiometers to reflect real-world frontline screening conditions.
3. **Household-Grouped Splitting**: NutriSense AI prevents sibling leakage via grouped household splitting on `(v001, v002)`, which was not explicitly controlled in Islam et al.
4. **Target-Specific Cohorts**: NutriSense AI preserves all valid records per target ($N > 200{,}000$) rather than dropping records with any missing target.
5. **Strict Three-Way Partitioning**: NutriSense AI establishes a locked Test set (15%) and tunes baselines exclusively on Validation (15%).

---

## 39. What Cannot Yet Be Claimed

- **DO NOT CLAIM**: "The baseline model is ready for clinical deployment."
- **DO NOT CLAIM**: "NutriSense AI outperforms Islam et al. (2024)."
- **DO NOT CLAIM**: "The model diagnoses childhood malnutrition."
- **CLAIM INSTEAD**: "Step 9 establishes a verified, leakage-safe linear baseline that provides moderate screening discriminability (ROC-AUC 0.62–0.67) and demonstrates the necessity of non-linear models and threshold calibration."

---

## 40. Reproducibility

- **Random Seed**: `SEED = 42`.
- Deterministic solvers (`lbfgs`) and fixed random states ensure identical coefficients and validation scores across runs.
- Model artifacts are serialized with `joblib` for version-controlled reproducibility.

---

## 41. Files Created

| File | Type | Purpose |
| :--- | :---: | :--- |
| [`src/models/baseline_model.py`](file:///d:/finalyearproj/Nutrisense-Ai/src/models/baseline_model.py) | Python Script | End-to-end training and validation script for Logistic Regression baselines. |
| [`models/baseline_logistic_stunting.joblib`](file:///d:/finalyearproj/Nutrisense-Ai/models/baseline_logistic_stunting.joblib) | Binary Artifact | Fitted Pipeline for Stunting. |
| [`models/baseline_logistic_underweight.joblib`](file:///d:/finalyearproj/Nutrisense-Ai/models/baseline_logistic_underweight.joblib) | Binary Artifact | Fitted Pipeline for Underweight. |
| [`models/baseline_logistic_wasting.joblib`](file:///d:/finalyearproj/Nutrisense-Ai/models/baseline_logistic_wasting.joblib) | Binary Artifact | Fitted Pipeline for Wasting. |
| [`data/interim/baseline_metrics.json`](file:///d:/finalyearproj/Nutrisense-Ai/data/interim/baseline_metrics.json) | JSON File | Structured machine-readable metrics for all three targets. |
| [`tests/test_baseline_model.py`](file:///d:/finalyearproj/Nutrisense-Ai/tests/test_baseline_model.py) | Test Suite | 7 automated test functions verifying Step 9 integrity. |
| [`docs/09_baseline_model.md`](file:///d:/finalyearproj/Nutrisense-Ai/docs/09_baseline_model.md) | Technical Documentation | Detailed 47-section report for viva and defense. |

---

## 42. Libraries Used and Why

- `scikit-learn`: Standard scientific ML library providing `Pipeline`, `ColumnTransformer`, `LogisticRegression`, and clinical evaluation metrics (`roc_auc_score`, `average_precision_score`).
- `pandas` & `numpy`: Efficient vectorized array and tabular data processing.
- `joblib`: High-efficiency serialization for scikit-learn pipelines with NumPy arrays.

---

## 43. Alternatives Considered

1. **Dummy / Majority Class Classifier**: Trivial baseline predicting 0 for all records. (Rejected as sole baseline because it lacks discriminative probabilities).
2. **Decision Tree Classifier (Single Tree)**: Simple non-linear baseline. (Deferred to Step 10 model comparison).
3. **Random Forests / Gradient Boosted Trees (LightGBM, XGBoost)**: Advanced non-linear models. (Reserved for Step 10).
4. **Target Encoding**: Supervised encoding of high-cardinality features. (Excluded in Step 9 to prevent target leakage risks in linear modeling).

---

## 44. Why Those Alternatives Are NOT Being Used Yet

A disciplined scientific methodology requires establishing the linear baseline first. Jumping directly to complex tree ensembles prevents researchers from quantifying the exact performance gain attributable to non-linear interactions versus simple log-odds relationships.

---

## 45. Viva Questions and Answers

**Q1: Why did you start with Logistic Regression rather than LightGBM or a Neural Network?**  
*Answer*: In clinical predictive modeling, Logistic Regression serves as the gold-standard reference benchmark. It establishes the baseline discriminative ability of the predictors under an additive linear log-odds formulation. Evaluating advanced tree models against this baseline reveals whether complex models provide true predictive gains or merely superfluous complexity.

**Q2: How did you ensure that preprocessing didn't leak validation or test information?**  
*Answer*: We used an `sklearn.pipeline.Pipeline` combining a `ColumnTransformer` and the classifier. When `pipeline.fit(X_train, y_train)` is called, the median imputer, standard scaler, and one-hot encoder compute summary statistics strictly from `X_train`. When evaluated on `X_val`, the pipeline applies those frozen training statistics without recalculating them. The test set was locked and never accessed.

**Q3: Why is the Recall for Wasting so low (0.07%) despite an Accuracy of 81.35%?**  
*Answer*: Wasting prevalence in the validation cohort is 18.65%. Because the classes are imbalanced and the linear model's predicted probabilities rarely exceed 0.50, the default decision threshold ($\tau = 0.5$) classifies 99.98% of children as negative. This yields high accuracy (since 81.35% truly are negative), but fails to screen positive cases. This mathematically demonstrates why Accuracy is deceptive in clinical screening and why ROC-AUC (0.6206) and PR-AUC (0.2596) are the appropriate ranking metrics.

**Q4: What is the difference between ROC-AUC and PR-AUC in this context?**  
*Answer*: ROC-AUC measures trade-offs across True Positive Rate and False Positive Rate, independent of class prevalence. PR-AUC measures Precision across Recall levels. In imbalanced datasets, ROC-AUC can present an overly optimistic view because large numbers of true negatives suppress the False Positive Rate; PR-AUC is more sensitive to false positive spikes and provides a rigorous evaluation of screening quality.

**Q5: Why did the feature space expand from 34 to 89 columns?**  
*Answer*: The 34 candidate features include 7 categorical/nominal predictors (e.g., `state_id`, `caste_category`, `religion_category`, `sanitation_facility_type`). The `OneHotEncoder` transforms these discrete categories into binary indicator columns (e.g., 36 columns for Indian states), resulting in an 89-dimensional design matrix.

**Q6: What is Scenario A and why does it matter for Step 9?**  
*Answer*: Scenario A models community-based pre-screening by frontline health workers (ASHAs) without physical height boards or weighing scales. All direct anthropometrics (`hw2`, `hw3`, `hw70`–`hw73`) were strictly excluded from $X$. The model predicts malnutrition risk solely from demographic, maternal, birth, and environmental risk factors.

**Q7: How did you handle target-specific cohorts during training?**  
*Answer*: NFHS-5 has target-specific validity ($N = 206{,}025$ for stunting, $N = 210{,}524$ for underweight, $N = 201{,}687$ for wasting). Each baseline model was trained strictly on observations where that specific target was valid, preventing sample truncation while keeping holdout households consistent.

**Q8: What does a Stunting ROC-AUC of 0.6607 mean practically?**  
*Answer*: An AUC of 0.6607 means that if an ASHA screens a randomly selected stunted child and a randomly selected non-stunted child, the model will assign a higher risk score to the stunted child approximately 66% of the time. This demonstrates meaningful pre-screening signal prior to physical anthropometry.

**Q9: Why didn't you apply SMOTE or oversampling in Step 9?**  
*Answer*: Step 9 is strictly a baseline. Resampling methods alter the natural posterior probability calibration and must be evaluated systematically in subsequent modeling steps alongside cost-sensitive loss functions and threshold tuning.

**Q10: Did you evaluate this model on the Test set?**  
*Answer*: No. Under the Test-Set Lock Principle, the test set ($N = 33{,}069$) remains untouched to prevent overfitting and evaluation snooping. The metrics reported are strictly from the Validation set.

---

## 46. Commands to Run Step 9

```powershell
# 1. Execute Step 9 baseline model training and validation
python src/models/baseline_model.py

# 2. Run Step 9 automated test suite
python tests/test_baseline_model.py

# 3. Run complete regression test suite (Steps 0 to 9)
python -c "
import subprocess, sys
scripts = [
    'tests/test_setup.py',
    'tests/test_inspection.py',
    'tests/test_data_dictionary.py',
    'tests/test_research_population.py',
    'tests/test_data_cleaning.py',
    'tests/test_target_creation.py',
    'tests/test_eda.py',
    'tests/test_feature_engineering.py',
    'tests/test_split_data.py',
    'tests/test_baseline_model.py'
]
for s in scripts:
    res = subprocess.run([sys.executable, s], capture_output=True, text=True)
    if res.returncode != 0:
        print(f'FAILED: {s}'); sys.exit(1)
    else:
        print(f'[PASS] {s}')
print('ALL TESTS PASSED!')
"
```

---

## 47. Next Step (Step 10 Preview)

**Step 10 — Advanced Model Comparison & Hyperparameter Exploration**:
- Introduce non-linear model architectures: **LightGBM**, **XGBoost**, **CatBoost**, and **Random Forest**.
- Systematically compare tree-based models against the Logistic Regression baseline on the Validation partition.
- Evaluate feature importance, class weighting (`class_weight='balanced'`), and non-linear interactions under Scenario A constraints.
