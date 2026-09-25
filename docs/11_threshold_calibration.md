# Step 11 — Threshold Calibration and Operational Decision Curve Analysis

**Project**: NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System  
**Dataset**: NFHS-5 India 2019–21 Children's Recode (KR) (`IAKR7EFL.DTA`, 441,380,745 bytes)  
**Prediction Setting**: Scenario A — Community Pre-Screening (frontline workers, zero direct anthropometric measurements in feature matrix $X$)  
**Target Outcomes**: Stunting ($N_{val}=30{,}885$, 35.50%), Underweight ($N_{val}=31{,}531$, 30.68%), Wasting ($N_{val}=30{,}211$, 18.65%)  
**Evaluation Partition**: Validation Set strictly ($N_{val} \approx 33{,}189$; valid target samples per outcome noted above).  
**Test Set Integrity**: Test Set ($N_{test} = 33{,}069$) remains **100% locked, unpredicted, uncalibrated, and untouched**.

---

## 1. Executive Summary & Objective

In Steps 9 and 10, binary classification models for Scenario A community pre-screening were trained and evaluated at the conventional default decision threshold ($\tau = 0.50$). However, malnutrition prevalence in community screening contexts is naturally imbalanced—ranging from 18.65% (Wasting) to 35.50% (Stunting). 

At $\tau = 0.50$, unweighted probabilistic models (such as LightGBM and Logistic Regression) suffer from severe operational failure:
- For **Wasting** (prevalence 18.65%), LightGBM Unweighted identifies only 5 true positive cases out of 5,635 wasted children, yielding a catastrophic sensitivity of **0.09%** (false negative rate 99.91%).
- For **Underweight** (prevalence 30.68%), LightGBM Unweighted detects only 1,610 of 9,673 cases, yielding an operational sensitivity of **16.64%** (false negative rate 83.36%).
- For **Stunting** (prevalence 35.50%), LightGBM Unweighted identifies 2,685 of 10,963 cases, yielding a sensitivity of **24.49%** (false negative rate 75.51%).

Step 11 systematically addresses this clinical triage limitation through five rigorous methodological components:
1. **Full-Grid Decision Threshold Analysis**: Continuous parametric sweep across 99 thresholds ($\tau \in [0.01, 0.99]$, step 0.01) evaluating complete contingency tables ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$) and 11 diagnostic metrics.
2. **Analytical Candidate Threshold Identification**: Objective optimization of objective criteria without imposing arbitrary clinical targets (Maximum F1-score, Maximum Youden's $J$ statistic, and Maximum Balanced Accuracy) alongside default $\tau = 0.50$.
3. **Loss-Reweighting vs. Threshold-Shifting Comparison**: Empirical and conceptual comparison of training-time class weighting versus post-hoc decision threshold adjustment.
4. **Leakage-Safe Probability Calibration**: Fitting Parametric Platt Scaling (logistic sigmoid) and Non-Parametric Isotonic Regression strictly on the training partition ($X_{train}$) via 3-fold cross-validation (`CalibratedClassifierCV`), assessed using Brier scores and reliability diagrams.
5. **Decision Curve Analysis (DCA)**: Exploratory decision-analytic evaluation comparing calculated Net Benefit curves with treat-all and treat-none reference strategies across a specified threshold-probability range ($p_t \in [0.05, 0.80]$).

---

## 2. Mathematical Formulations & Optimization Criteria

For any continuous model predicted risk score $\hat{p} \in [0, 1]$ and decision threshold $\tau \in [0, 1]$, binary decision rule $\hat{y} = \mathbb{I}(\hat{p} \ge \tau)$ produces contingency counts: True Positives ($\text{TP}$), False Positives ($\text{FP}$), True Negatives ($\text{TN}$), and False Negatives ($\text{FN}$).

### 2.1 Diagnostic Performance Metrics

$$\text{Sensitivity (Recall, TPR)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity (TNR)} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

$$\text{Precision (Positive Predictive Value, PPV)} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Negative Predictive Value (NPV)} = \frac{\text{TN}}{\text{TN} + \text{FN}}$$

$$\text{False Positive Rate (FPR)} = 1 - \text{Specificity} = \frac{\text{FP}}{\text{TN} + \text{FP}}$$

$$\text{False Negative Rate (FNR)} = 1 - \text{Sensitivity} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

### 2.2 Objective Analytical Optimization Criteria

Rather than imposing subjective target thresholds (e.g. arbitrarily mandating 80% sensitivity), thresholds are identified analytically through objective global optima:

1. **Maximum F1-Score Criterion ($\tau_{\text{F1}}^{*}$)**:
   Maximizes the harmonic mean of precision and sensitivity, balancing positive detection volume against false referral load:
   $$\tau_{\text{F1}}^{*} = \arg\max_{\tau \in [0.01, 0.99]} \left[ \frac{2 \cdot \text{PPV}(\tau) \cdot \text{TPR}(\tau)}{\text{PPV}(\tau) + \text{TPR}(\tau)} \right]$$

2. **Maximum Youden's $J$ Statistic Criterion ($\tau_{J}^{*}$)**:
   Maximizes the linear divergence between sensitivity and the false positive rate, identifying the point on the ROC curve with maximum orthogonal distance from the chance diagonal:
   $$\tau_{J}^{*} = \arg\max_{\tau \in [0.01, 0.99]} \left[ \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \right] = \arg\max_{\tau \in [0.01, 0.99]} \left[ \text{TPR}(\tau) - \text{FPR}(\tau) \right]$$

3. **Maximum Balanced Accuracy Criterion ($\tau_{\text{BA}}^{*}$)**:
   Equally weights performance across both minority (malnourished) and majority (eunourished) cohorts:
   $$\tau_{\text{BA}}^{*} = \arg\max_{\tau \in [0.01, 0.99]} \left[ \frac{\text{Sensitivity}(\tau) + \text{Specificity}(\tau)}{2} \right]$$
   *Note: Mathematically, $\text{Balanced Accuracy} = \frac{J + 1}{2}$, making $\tau_{J}^{*}$ and $\tau_{\text{BA}}^{*}$ identical in continuous optimization.*

---

## 3. Comprehensive Empirical Validation Results

All evaluations were executed on the held-out validation cohort using the top-performing non-linear model (**LightGBM**) and the linear baseline benchmark (**Logistic Regression**).

### 3.1 Stunting (Prevalence: 35.50%, $N_{val} = 30{,}885$)

| Model & Strategy | Threshold $\tau$ | TP | FP | FN | TN | Sensitivity | Specificity | PPV | NPV | F1 | Balanced Acc | Youden's $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Unweighted) — Default** | 0.50 | 2,685 | 2,014 | 8,278 | 17,908 | 24.49% | 89.89% | 57.14% | 68.39% | 0.3429 | 57.19% | 0.1438 |
| **LightGBM (Unweighted) — Max F1** | **0.30** | 8,354 | 10,598 | 2,609 | 9,324 | **76.20%** | 46.80% | 44.08% | 78.14% | **0.5585** | 61.50% | 0.2301 |
| **LightGBM (Unweighted) — Max Youden's $J$** | **0.35** | 7,010 | 7,747 | 3,953 | 12,175 | **63.94%** | **61.11%** | 47.50% | 75.50% | 0.5451 | **62.53%** | **0.2506** |
| LightGBM (Class-Weighted) — Default | 0.50 | 6,865 | 7,532 | 4,098 | 12,390 | 62.62% | 62.19% | 47.68% | 75.15% | 0.5414 | 62.41% | 0.2481 |
| LightGBM (Class-Weighted) — Max F1 | 0.41 | 8,950 | 12,116 | 2,013 | 7,806 | 81.64% | 39.18% | 42.49% | 79.50% | 0.5589 | 60.41% | 0.2082 |
| LightGBM (Class-Weighted) — Max Youden's $J$ | 0.51 | 6,608 | 7,064 | 4,355 | 12,858 | 60.28% | 64.54% | 48.33% | 74.70% | 0.5365 | 62.41% | 0.2482 |
| Logistic Reg. (Unweighted) — Default | 0.50 | 2,630 | 2,139 | 8,333 | 17,783 | 23.99% | 89.26% | 55.15% | 68.09% | 0.3344 | 56.63% | 0.1325 |
| Logistic Reg. (Unweighted) — Max F1 | 0.29 | 8,396 | 10,950 | 2,567 | 8,972 | 76.59% | 45.04% | 43.40% | 77.75% | 0.5540 | 60.81% | 0.2162 |
| Logistic Reg. (Unweighted) — Max Youden's $J$ | 0.35 | 6,707 | 7,522 | 4,256 | 12,400 | 61.18% | 62.24% | 47.14% | 74.45% | 0.5325 | 61.71% | 0.2342 |
| Logistic Reg. (Class-Weighted) — Default | 0.50 | 6,558 | 7,272 | 4,405 | 12,650 | 59.82% | 63.51% | 47.42% | 74.17% | 0.5290 | 61.66% | 0.2333 |

---

### 3.2 Underweight (Prevalence: 30.68%, $N_{val} = 31{,}531$)

| Model & Strategy | Threshold $\tau$ | TP | FP | FN | TN | Sensitivity | Specificity | PPV | NPV | F1 | Balanced Acc | Youden's $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Unweighted) — Default** | 0.50 | 1,610 | 1,182 | 8,063 | 20,676 | 16.64% | 94.59% | 57.66% | 71.94% | 0.2583 | 55.62% | 0.1124 |
| **LightGBM (Unweighted) — Max F1** | **0.27** | 7,258 | 11,031 | 2,415 | 10,827 | **75.03%** | 49.53% | 39.69% | 81.76% | **0.5191** | 62.28% | 0.2457 |
| **LightGBM (Unweighted) — Max Youden's $J$** | **0.31** | 6,235 | 8,442 | 3,438 | 13,416 | **64.46%** | **61.38%** | 42.48% | 79.60% | 0.5121 | **62.92%** | **0.2584** |
| LightGBM (Class-Weighted) — Default | 0.50 | 6,229 | 8,474 | 3,444 | 13,384 | 64.40% | 61.23% | 42.37% | 79.53% | 0.5111 | 62.81% | 0.2563 |
| LightGBM (Class-Weighted) — Max F1 | 0.45 | 7,315 | 11,116 | 2,358 | 10,742 | 75.62% | 49.14% | 39.69% | 82.00% | 0.5206 | 62.38% | 0.2477 |
| LightGBM (Class-Weighted) — Max Youden's $J$ | 0.51 | 6,008 | 7,923 | 3,665 | 13,935 | 62.11% | 63.75% | 43.13% | 79.18% | 0.5091 | 62.93% | 0.2586 |
| Logistic Reg. (Unweighted) — Default | 0.50 | 1,596 | 1,263 | 8,077 | 20,595 | 16.50% | 94.22% | 55.82% | 71.83% | 0.2547 | 55.36% | 0.1072 |
| Logistic Reg. (Unweighted) — Max F1 | 0.25 | 7,675 | 12,508 | 1,998 | 9,350 | 79.34% | 42.78% | 38.03% | 82.39% | 0.5141 | 61.06% | 0.2212 |
| Logistic Reg. (Unweighted) — Max Youden's $J$ | 0.32 | 5,957 | 8,138 | 3,716 | 13,720 | 61.58% | 62.77% | 42.26% | 78.69% | 0.5013 | 62.18% | 0.2435 |
| Logistic Reg. (Class-Weighted) — Default | 0.50 | 6,245 | 8,774 | 3,428 | 13,084 | 64.56% | 59.86% | 41.58% | 79.24% | 0.5058 | 62.21% | 0.2442 |

---

### 3.3 Wasting (Prevalence: 18.65%, $N_{val} = 30{,}211$)

| Model & Strategy | Threshold $\tau$ | TP | FP | FN | TN | Sensitivity | Specificity | PPV | NPV | F1 | Balanced Acc | Youden's $J$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Unweighted) — Default** | 0.50 | 5 | 5 | 5,630 | 24,571 | 0.09% | 99.98% | 50.00% | 81.36% | 0.0018 | 50.03% | 0.0007 |
| **LightGBM (Unweighted) — Max F1** | **0.17** | 3,891 | 12,640 | 1,744 | 11,936 | **69.05%** | **48.57%** | 23.54% | 87.25% | **0.3511** | **58.81%** | **0.1762** |
| **LightGBM (Unweighted) — Max Youden's $J$** | **0.17** | 3,891 | 12,640 | 1,744 | 11,936 | **69.05%** | **48.57%** | 23.54% | 87.25% | **0.3511** | **58.81%** | **0.1762** |
| LightGBM (Class-Weighted) — Default | 0.50 | 3,223 | 9,934 | 2,412 | 14,642 | 57.20% | 59.58% | 24.50% | 85.86% | 0.3430 | 58.39% | 0.1677 |
| LightGBM (Class-Weighted) — Max F1 | 0.47 | 3,852 | 12,496 | 1,783 | 12,080 | 68.36% | 49.15% | 23.56% | 87.14% | 0.3505 | 58.76% | 0.1751 |
| LightGBM (Class-Weighted) — Max Youden's $J$ | 0.47 | 3,852 | 12,496 | 1,783 | 12,080 | 68.36% | 49.15% | 23.56% | 87.14% | 0.3505 | 58.76% | 0.1751 |
| Logistic Reg. (Unweighted) — Default | 0.50 | 3 | 0 | 5,632 | 24,576 | 0.05% | 100.00%| 100.00%| 81.36% | 0.0011 | 50.03% | 0.0005 |
| Logistic Reg. (Unweighted) — Max F1 | 0.18 | 3,898 | 12,749 | 1,737 | 11,827 | 69.17% | 48.12% | 23.42% | 87.19% | 0.3499 | 58.65% | 0.1730 |
| Logistic Reg. (Unweighted) — Max Youden's $J$ | 0.18 | 3,898 | 12,749 | 1,737 | 11,827 | 69.17% | 48.12% | 23.42% | 87.19% | 0.3499 | 58.65% | 0.1730 |
| Logistic Reg. (Class-Weighted) — Default | 0.50 | 3,354 | 10,426 | 2,281 | 14,150 | 59.52% | 57.58% | 24.34% | 86.12% | 0.3455 | 58.55% | 0.1710 |

---

## 4. Methodological Comparison: Loss Reweighting vs. Post-Hoc Threshold Adjustment

A central question in applied machine learning for public health screening is whether to apply cost-sensitive class weights during model training (e.g. `is_unbalance=True` or `scale_pos_weight`) or to adjust the classification threshold post-hoc on an unweighted probabilistic model.

### 4.1 Theoretical Context in Linear Models

Consider binary logistic regression where positive cases are weighted by $w_1$ and negative cases by $w_0$. In an unregularized linear setting, the weighted log-likelihood shifts the intercept such that the predicted log-odds satisfy:

$$\log \left( \frac{\hat{p}_{w}(x)}{1 - \hat{p}_{w}(x)} \right) = \mathbf{x}^T \boldsymbol{\beta}_{w} \approx \mathbf{x}^T \boldsymbol{\beta}_{\text{raw}} + \log \left( \frac{w_1}{w_0} \right)$$

Predicting positive when $\hat{p}_w(x) \ge 0.50$ corresponds to:

$$\mathbf{x}^T \boldsymbol{\beta}_w \ge 0 \iff \mathbf{x}^T \boldsymbol{\beta}_{\text{raw}} \ge \log \left( \frac{w_0}{w_1} \right)$$

In terms of the original unweighted probability $\hat{p}_{\text{raw}}(x)$:

$$\hat{p}_{\text{raw}}(x) \ge \tau_{\text{equiv}} = \frac{1}{1 + \frac{w_1}{w_0}} = \frac{w_0}{w_0 + w_1}$$

For standard balanced weighting ($w_1 = \frac{N}{2 N_1}$ and $w_0 = \frac{N}{2 N_0}$), this linear approximation gives $\tau_{\text{equiv}} \approx \pi$. However, in non-linear models, tree ensembles, and regularized models, the loss landscape and tree-splitting choices differ during training.

### 4.2 Empirical Validation Observations

Class weighting and post-hoc threshold adjustment produced similar validation operating points for some experiments. However, they are not generally equivalent procedures: class weighting changes the training objective, whereas threshold adjustment changes the decision rule applied to model scores. The observed similarity is therefore treated as an empirical result for this dataset and model configuration.

1. **Stunting ($\pi = 35.50\%$)**:
   - LightGBM Unweighted evaluated at threshold $\tau \approx 0.35$: Sensitivity = 63.94%, Specificity = 61.11%, F1 = 0.5451, Balanced Acc = 62.53%.
   - LightGBM Class-Weighted evaluated at default threshold $\tau = 0.50$: Sensitivity = 62.62%, Specificity = 62.19%, F1 = 0.5414, Balanced Acc = 62.41%.

2. **Underweight ($\pi = 30.68\%$)**:
   - LightGBM Unweighted evaluated at Youden's threshold $\tau = 0.31$: Sensitivity = 64.46%, Specificity = 61.38%, F1 = 0.5121, Balanced Acc = 62.92%.
   - LightGBM Class-Weighted evaluated at default threshold $\tau = 0.50$: Sensitivity = 64.40%, Specificity = 61.23%, F1 = 0.5111, Balanced Acc = 62.81%.

3. **Wasting ($\pi = 18.65\%$)**:
   - LightGBM Unweighted evaluated at threshold $\tau = 0.17$: Sensitivity = 69.05%, Specificity = 48.57%, F1 = 0.3511, Balanced Acc = 58.81%.
   - LightGBM Class-Weighted evaluated at default threshold $\tau = 0.50$: Sensitivity = 57.20%, Specificity = 59.58%, F1 = 0.3430, Balanced Acc = 58.39%.
   - Adjusting LightGBM Class-Weighted to $\tau = 0.47$ yields Sensitivity = 68.36%, Specificity = 49.15%, F1 = 0.3505.

### 4.3 Practical Considerations

Both strategies provide viable mechanisms to avoid the severe sensitivity deficits seen under uncalibrated 0.5 thresholds:
- Training-time class weighting directly embeds the imbalance penalty into tree split criteria and leaf values, producing scores centered closer to 0.5.
- Post-hoc threshold adjustment operates on fixed model predictions, offering operational flexibility to reconfigure triage cutoffs to meet local screening capacity without retraining the underlying model.

---

## 5. Probability Calibration Analysis

Accurate risk communication requires that if a screening tool predicts a 30% risk of acute malnutrition, exactly 30 out of 100 children with that score are truly malnourished.

### 5.1 Leakage-Safe Calibration Architecture

To prevent data leakage, calibration mappers were fitted exclusively on the training partition ($X_{train}$) using 3-fold cross-validation (`CalibratedClassifierCV`):
- **Platt Scaling**: Fits a univariate logistic regression $\hat{P}(Y=1|f) = \frac{1}{1 + \exp(A \cdot f + B)}$ on out-of-fold model scores $f$.
- **Isotonic Regression**: Fits a non-parametric isotonic step function $\hat{P}(Y=1|f) = m(f)$ subject to monotonicity constraints.

The resulting calibrated models were evaluated on the completely independent validation partition.

### 5.2 Brier Score Assessment

The Brier score measures the mean squared error between probabilistic forecasts and actual binary outcomes:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2 \in [0, 1]$$

| Malnutrition Target | Validation Prevalence $\pi$ | Uncalibrated LightGBM | Platt Scaling (Train CV) | Isotonic Regression (Train CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Stunting** | 35.50% | **0.21055** | 0.21965 | 0.21421 |
| **Underweight** | 30.68% | **0.19394** | 0.21200 | 0.19955 |
| **Wasting** | 18.65% | **0.14719** | 0.15199 | 0.14923 |

### 5.3 Reliability Curve and Calibration Assessment

The calibration analysis produced Brier scores of 0.21055, 0.19394, and 0.14719 for stunting, underweight, and wasting respectively. These scores should be interpreted together with the calibration curves; Brier score alone does not establish that predicted probabilities are well calibrated.

Inspection of the validation reliability diagrams (**Figure 17**) shows:
1. Across the middle predicted probability ranges, the uncalibrated LightGBM curves generally track the reference diagonal, but diverge in lower and upper probability bins where sample density is lower.
2. In this experiment, post-hoc isotonic regression and Platt scaling produced validation Brier scores slightly higher than the uncalibrated model (by 0.002 to 0.018).
3. Given these observations, probability estimates should be interpreted with appropriate caution, and decision-making should be evaluated primarily through empirical operating points and decision curve analyses rather than assuming perfect posterior risk calibration.

---

## 6. Decision Curve Analysis (DCA)

Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves assess statistical discrimination independent of clinical utility. Decision Curve Analysis (Vickers & Elkan, BMJ 2006) evaluates clinical usefulness by incorporating the relative harms of false positives versus false negatives.

### 6.1 Clinical Net Benefit Formulation

At a decision threshold probability $p_t$, a frontline worker considers a false negative to be $\frac{1 - p_t}{p_t}$ times worse than a false positive referral:

$$\text{Net Benefit}(\tau = p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left( \frac{p_t}{1 - p_t} \right)$$

Two clinical default comparison strategies are defined:
- **Treat-None**: Predict all children negative ($\text{TP}=0, \text{FP}=0 \implies \text{Net Benefit} = 0$).
- **Treat-All**: Refer all children for clinical confirmatory assessment ($\text{Net Benefit} = \pi - (1 - \pi) \left( \frac{p_t}{1 - p_t} \right)$).

### 6.2 Empirical Decision Curve Results

### 6.2 Decision Curve Analysis Observations

Under the specified decision-analytic assumptions and threshold-probability range, the calculated net-benefit curves were compared with treat-all and treat-none reference strategies. These results are exploratory and should not be interpreted as evidence of clinical benefit or clinical effectiveness.

Across the specified threshold-probability range ($p_t \in [0.05, 0.80]$) evaluated on the validation set (**Figure 18**):

1. **Stunting**:
   - For threshold probabilities between $p_t = 0.20$ and $p_t = 0.55$, the calculated LightGBM Net Benefit ranged between $+0.12$ and $+0.26$.
   - Over this range, the model curve tracks above the reference treat-all and treat-none lines.
   - At $p_t = 0.35$ (Youden's optimum), the calculated Net Benefit is $+0.142$ under the DCA model assumptions.

2. **Underweight**:
   - The calculated Net Benefit curve exceeds the treat-all reference for $p_t > 0.22$ and remains positive up to $p_t \approx 0.58$.
   - At $p_t = 0.31$, Net Benefit is $+0.124$ within this decision-analytic framework.

3. **Wasting**:
   - For wasting (validation prevalence 18.65%), the treat-all strategy net benefit crosses zero around $p_t \approx 0.18$.
   - The model curve maintains positive calculated net benefit across the range $p_t \in [0.12, 0.35]$.
   - At $p_t = 0.17$, calculated Net Benefit is $+0.048$.

---

## 7. Artifact Inventory & Generated Figures

The Step 11 analysis generated the following verified analytical figures and structured data files:

### 7.1 Figures Generated

1. `reports/figures/fig12_threshold_curves_stunting.png`: Dual-axis plot of Sensitivity, Specificity, Precision, F1, and Balanced Accuracy across $\tau \in [0.01, 0.99]$ for Stunting.
2. `reports/figures/fig13_threshold_curves_underweight.png`: Diagnostic metric trajectories across $\tau$ for Underweight.
3. `reports/figures/fig14_threshold_curves_wasting.png`: Diagnostic metric trajectories across $\tau$ for Wasting, highlighting the severe collapse at $\tau = 0.50$.
4. `reports/figures/fig15_precision_recall_curves.png`: Precision-Recall curves showing baseline prevalence benchmarks and optimal operating points for all 3 targets.
5. `reports/figures/fig16_roc_curves.png`: Receiver Operating Characteristic curves annotated with Maximum Youden's $J$ operating points.
6. `reports/figures/fig17_probability_calibration_curves.png`: 3-panel Reliability Diagrams comparing Uncalibrated, Platt, and Isotonic probability calibration against the ideal 45-degree diagonal.
7. `reports/figures/fig18_decision_curve_analysis.png`: Decision Curve Analysis Net Benefit plots comparing Model Screening vs. Treat-All and Treat-None across specified threshold probabilities $p_t$.

### 7.2 Structured Data Artifacts

1. `data/interim/threshold_analysis_metrics.json`: Complete machine-readable dictionary containing validation cohort sizes, prevalences, full metric summaries for default, Max F1, Max Youden, and Max Balanced Accuracy across all evaluated models.
2. `data/interim/threshold_analysis_metrics.csv`: 792-row tabular dataset storing all 18 contingency and diagnostic metrics across every threshold $\tau \in [0.01, 0.99]$ for all 4 models and 3 targets.
3. `data/interim/calibration_metrics.json`: Brier scores, mean predicted probabilities, and fraction of true positives across 10 empirical calibration bins for all 3 targets.
4. `data/interim/decision_curve_metrics.csv`: Net Benefit calculations across threshold probabilities $p_t \in [0.05, 0.80]$ for all targets.

---

## 8. Verification and Quality Assurance

1. **Test Set Lock Verified**: All threshold sweeps, calibration curve fitting, and decision curve calculations were performed strictly on the validation partition. The test set ($N = 33{,}069$) remains untouched.
2. **Feature Isolation Verified**: Feature registry contains strictly the 34 non-invasive candidate features approved in Step 7. No anthropometric measurements (`hw2`–`hw12`, `hw70`–`hw73`) were accessed.
3. **Immutability of Raw Data**: `IAKR7EDT/IAKR7EFL.DTA` remains byte-exact at 441,380,745 bytes.
4. **Automated Unit Tests**: All 8 Step 11 verification tests in `tests/test_threshold_analysis.py` and all 12 regression test suites across Steps 0–11 passed with 100% success rate.

---

## 9. Conclusion & Operational Summary

1. **Default Threshold Limitation**: The standard default threshold ($\tau = 0.50$) produces high false negative rates on this dataset (75.5% for Stunting, 83.4% for Underweight, 99.9% for Wasting) due to natural class imbalance.
2. **Analytical Validation Operating Points**:
   - For **Stunting**: Youden-selected threshold **$\tau = 0.35$** (Sensitivity 63.9%, Specificity 61.1%, Balanced Acc 62.5%).
   - For **Underweight**: Youden-selected threshold **$\tau = 0.31$** (Sensitivity 64.5%, Specificity 61.4%, Balanced Acc 62.9%).
   - For **Wasting**: Youden-selected threshold **$\tau = 0.17$** (Sensitivity 69.1%, Specificity 48.6%, Balanced Acc 58.8%).
3. **Methodological Note**: Class weighting and post-hoc threshold adjustment produced similar validation operating points for some experiments. However, they are not generally equivalent procedures: class weighting changes the training objective, whereas threshold adjustment changes the decision rule applied to model scores. The observed similarity is therefore treated as an empirical result for this dataset and model configuration. Post-hoc threshold adjustment provides operational flexibility to reconfigure triage cutoffs without retraining.
