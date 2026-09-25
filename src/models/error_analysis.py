#!/usr/bin/env python3
"""
Step 15: Descriptive Error Analysis & Subgroup Analysis
Project: NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence
Dataset: NFHS-5 India 2019–21 Children's Recode (KR)
Setting: Scenario A — Community Pre-Screening (34 non-invasive features)

Performs rigorous descriptive error analysis and subgroup auditing using the
locked validation cohort predictions and pre-selected champion LightGBM Unweighted models.

CRITICAL METHODOLOGICAL RULES:
1. Purely descriptive: zero model retraining, zero hyperparameter tuning,
   zero feature selection, and zero threshold re-optimization.
2. Evaluated strictly on the VALIDATION partition (split == 'val').
   The locked test partition remains untouched.
3. Decision thresholds locked at validation-derived values:
   - Stunting: tau = 0.35
   - Underweight: tau = 0.31
   - Wasting: tau = 0.17
4. Subgroup variables are restricted to approved Step 7 Scenario A features:
   child_age_group, child_sex_male, wealth_quintile, mother_education_level,
   is_rural, recent morbidity (diarrhea, fever, cough), and state_id.
5. Zero leakage: no direct anthropometric variables (hw2-hw12, hw70-hw73, hw13).
6. Minimum sample rule: N >= 200 for comparative reporting; smaller groups flagged as descriptive.
7. state_id is an administrative geographic code (one-hot encoded in the models);
   no continuous or ordinal interpretations.
8. Raw DHS microdata byte count verified (441,380,745 bytes). Zero individual microdata exported.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.features.build_features import transform_candidate_features

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"

OUTPUT_METRICS_CSV = "data/interim/error_analysis_metrics.csv"
OUTPUT_METRICS_JSON = "data/interim/error_analysis_metrics.json"
OUTPUT_OVERLAP_JSON = "data/interim/error_overlap_metrics.json"

RANDOM_SEED = 42
N_BOOTSTRAP = 500
MIN_SUBGROUP_N = 200

TARGET_CONFIGS = {
    "stunting": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_stunting_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.35,
        "eligibility_col": "eligible_stunting"
    },
    "underweight": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_underweight_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.31,
        "eligibility_col": "eligible_underweight"
    },
    "wasting": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_wasting_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.17,
        "eligibility_col": "eligible_wasting"
    }
}

SUBGROUP_DIMENSIONS = [
    {
        "name": "child_age_group",
        "feature": "child_age_group",
        "display_name": "Child Age Group",
        "order": ["00_05_mo", "06_11_mo", "12_23_mo", "24_35_mo", "36_47_mo", "48_59_mo"],
        "labels": {
            "00_05_mo": "0-5 months",
            "06_11_mo": "6-11 months",
            "12_23_mo": "12-23 months",
            "24_35_mo": "24-35 months",
            "36_47_mo": "36-47 months",
            "48_59_mo": "48-59 months"
        }
    },
    {
        "name": "child_sex",
        "feature": "child_sex_male",
        "display_name": "Child Sex",
        "order": [1, 0],
        "labels": {1: "Male", 0: "Female"}
    },
    {
        "name": "wealth_quintile",
        "feature": "wealth_quintile",
        "display_name": "Household Wealth Quintile",
        "order": [1.0, 2.0, 3.0, 4.0, 5.0],
        "labels": {1.0: "1 - Poorest", 2.0: "2 - Poorer", 3.0: "3 - Middle", 4.0: "4 - Richer", 5.0: "5 - Richest"}
    },
    {
        "name": "maternal_education",
        "feature": "mother_education_level",
        "display_name": "Maternal Education Level",
        "order": [0.0, 1.0, 2.0, 3.0],
        "labels": {0.0: "No education", 1.0: "Primary", 2.0: "Secondary", 3.0: "Higher"}
    },
    {
        "name": "residence_type",
        "feature": "is_rural",
        "display_name": "Place of Residence",
        "order": [1, 0],
        "labels": {1: "Rural", 0: "Urban"}
    },
    {
        "name": "diarrhea_recent",
        "feature": "diarrhea_recent",
        "display_name": "Recent Diarrhea (Past 2 Weeks)",
        "order": [1.0, 0.0],
        "labels": {1.0: "Yes", 0.0: "No"}
    },
    {
        "name": "fever_recent",
        "feature": "fever_recent",
        "display_name": "Recent Fever (Past 2 Weeks)",
        "order": [1.0, 0.0],
        "labels": {1.0: "Yes", 0.0: "No"}
    },
    {
        "name": "cough_recent",
        "feature": "cough_recent",
        "display_name": "Recent Cough (Past 2 Weeks)",
        "order": [1.0, 0.0],
        "labels": {1.0: "Yes", 0.0: "No"}
    },
    {
        "name": "state_ut",
        "feature": "state_id",
        "display_name": "State / Union Territory",
        "order": list(range(1, 37)),
        "labels": {s: f"State/UT {s:02d}" for s in range(1, 37)}
    }
]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def verify_raw_dataset_immutability():
    """Verify raw DHS data byte size integrity."""
    if not os.path.exists(RAW_DTA_PATH):
        raise FileNotFoundError(f"Raw dataset {RAW_DTA_PATH} not found.")
    actual_bytes = os.path.getsize(RAW_DTA_PATH)
    if actual_bytes != EXPECTED_RAW_BYTES:
        raise ValueError(f"Raw dataset altered! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}.")
    print(f"[OK] Raw dataset immutability verified ({actual_bytes:,} bytes).")


def compute_contingency_metrics(y_true, y_pred):
    """Computes exact contingency counts and diagnostic rates."""
    n = len(y_true)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n_pos = tp + fn
    n_neg = tn + fp

    prev = n_pos / n if n > 0 else 0.0
    sens = tp / n_pos if n_pos > 0 else np.nan
    spec = tn / n_neg if n_neg > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fnr = fn / n_pos if n_pos > 0 else np.nan
    fpr = fp / n_neg if n_neg > 0 else np.nan

    if not np.isnan(ppv) and not np.isnan(sens) and (ppv + sens) > 0:
        f1 = (2.0 * ppv * sens) / (ppv + sens)
    else:
        f1 = np.nan

    return {
        "sample_size": n,
        "actual_positives": n_pos,
        "actual_negatives": n_neg,
        "prevalence": prev,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "f1_score": f1
    }


def compute_bootstrap_cis(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Computes non-parametric 95% bootstrap confidence intervals for key rates."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    boot_sens = []
    boot_spec = []
    boot_ppv = []
    boot_f1 = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_y = y_true[idx]
        b_p = y_pred[idx]

        tp = np.sum((b_y == 1) & (b_p == 1))
        fp = np.sum((b_y == 0) & (b_p == 1))
        tn = np.sum((b_y == 0) & (b_p == 0))
        fn = np.sum((b_y == 1) & (b_p == 0))

        if (tp + fn) > 0:
            sens = tp / (tp + fn)
            boot_sens.append(sens)
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
            boot_spec.append(spec)
        if (tp + fp) > 0:
            ppv = tp / (tp + fp)
            boot_ppv.append(ppv)
            if (tp + fn) > 0 and (ppv + sens) > 0:
                f1 = (2.0 * ppv * sens) / (ppv + sens)
                boot_f1.append(f1)

    def extract_ci(vals):
        if len(vals) < 50:
            return {"ci_lower": None, "ci_upper": None}
        return {
            "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper": round(float(np.percentile(vals, 97.5)), 4)
        }

    return {
        "sensitivity_95ci": extract_ci(boot_sens),
        "specificity_95ci": extract_ci(boot_spec),
        "ppv_95ci": extract_ci(boot_ppv),
        "f1_score_95ci": extract_ci(boot_f1)
    }


def generate_error_visualizations(df_metrics):
    """Generates the 6 required publication-quality figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    palette_targets = {"stunting": "#3182ce", "underweight": "#38a169", "wasting": "#dd6b20"}

    # 1. Validation Sensitivity by Age Group (Fig 33)
    df_age = df_metrics[df_metrics["dimension"] == "child_age_group"].copy()
    age_labels_order = ["0-5 months", "6-11 months", "12-23 months", "24-35 months", "36-47 months", "48-59 months"]
    df_age["category_label"] = pd.Categorical(df_age["category_label"], categories=age_labels_order, ordered=True)
    df_age = df_age.sort_values("category_label")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_age, x="category_label", y="sensitivity", hue="target", palette=palette_targets, edgecolor="black", linewidth=0.5)
    plt.title("Validation Screening Sensitivity by Child Age Group", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Child Age Group", fontsize=11, fontweight="bold")
    plt.ylabel("Sensitivity (Recall)", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(title="Target Outcome", frameon=True)
    plt.tight_layout()
    fig33_path = os.path.join(FIGURES_DIR, "33_error_sensitivity_by_age.png")
    plt.savefig(fig33_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Fig 33: {fig33_path}")

    # 2. Validation Specificity by Age Group (Fig 34)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_age, x="category_label", y="specificity", hue="target", palette=palette_targets, edgecolor="black", linewidth=0.5)
    plt.title("Validation Screening Specificity by Child Age Group", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Child Age Group", fontsize=11, fontweight="bold")
    plt.ylabel("Specificity", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(title="Target Outcome", frameon=True)
    plt.tight_layout()
    fig34_path = os.path.join(FIGURES_DIR, "34_error_specificity_by_age.png")
    plt.savefig(fig34_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Fig 34: {fig34_path}")

    # 3. Validation F1 by Wealth Quintile (Fig 35)
    df_wealth = df_metrics[df_metrics["dimension"] == "wealth_quintile"].copy()
    wealth_labels_order = ["1 - Poorest", "2 - Poorer", "3 - Middle", "4 - Richer", "5 - Richest"]
    df_wealth["category_label"] = pd.Categorical(df_wealth["category_label"], categories=wealth_labels_order, ordered=True)
    df_wealth = df_wealth.sort_values("category_label")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_wealth, x="category_label", y="f1_score", hue="target", palette=palette_targets, edgecolor="black", linewidth=0.5)
    plt.title("Validation F1 Score by Household Wealth Quintile", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Household Wealth Quintile", fontsize=11, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=11, fontweight="bold")
    plt.ylim(0, 0.8)
    plt.legend(title="Target Outcome", frameon=True)
    plt.tight_layout()
    fig35_path = os.path.join(FIGURES_DIR, "35_error_f1_by_wealth.png")
    plt.savefig(fig35_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Fig 35: {fig35_path}")

    # 4. False-Negative Rate by Wealth Quintile (Fig 36)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_wealth, x="category_label", y="false_negative_rate", hue="target", palette=palette_targets, edgecolor="black", linewidth=0.5)
    plt.title("Validation False-Negative Rate (FNR) by Household Wealth Quintile", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Household Wealth Quintile", fontsize=11, fontweight="bold")
    plt.ylabel("False-Negative Rate (1 - Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylim(0, 0.8)
    plt.legend(title="Target Outcome", frameon=True)
    plt.tight_layout()
    fig36_path = os.path.join(FIGURES_DIR, "36_false_negative_rate_by_wealth.png")
    plt.savefig(fig36_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Fig 36: {fig36_path}")

    # 5. False-Negative Rate by Maternal Education (Fig 37)
    df_edu = df_metrics[df_metrics["dimension"] == "maternal_education"].copy()
    edu_labels_order = ["No education", "Primary", "Secondary", "Higher"]
    df_edu["category_label"] = pd.Categorical(df_edu["category_label"], categories=edu_labels_order, ordered=True)
    df_edu = df_edu.sort_values("category_label")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_edu, x="category_label", y="false_negative_rate", hue="target", palette=palette_targets, edgecolor="black", linewidth=0.5)
    plt.title("Validation False-Negative Rate by Maternal Education Level", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Maternal Education Level", fontsize=11, fontweight="bold")
    plt.ylabel("False-Negative Rate (1 - Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylim(0, 0.8)
    plt.legend(title="Target Outcome", frameon=True)
    plt.tight_layout()
    fig37_path = os.path.join(FIGURES_DIR, "37_false_negative_rate_by_education.png")
    plt.savefig(fig37_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Fig 37: {fig37_path}")

    # 6. Error-Type Distribution Across Major Age Groups (Fig 38)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    targets = ["stunting", "underweight", "wasting"]

    for idx, target in enumerate(targets):
        sub_t = df_age[df_age["target"] == target].copy().sort_values("category_label")
        categories = sub_t["category_label"].tolist()
        
        # Proportions of total N in each error quadrant
        tp_pct = sub_t["tp"] / sub_t["sample_size"] * 100
        fp_pct = sub_t["fp"] / sub_t["sample_size"] * 100
        tn_pct = sub_t["tn"] / sub_t["sample_size"] * 100
        fn_pct = sub_t["fn"] / sub_t["sample_size"] * 100

        ax = axes[idx]
        x_pos = np.arange(len(categories))
        ax.bar(x_pos, tp_pct, label="True Positive (TP)", color="#2b6cb0", edgecolor="white")
        ax.bar(x_pos, fn_pct, bottom=tp_pct, label="False Negative (FN)", color="#e53e3e", edgecolor="white")
        ax.bar(x_pos, fp_pct, bottom=tp_pct + fn_pct, label="False Positive (FP)", color="#dd6b20", edgecolor="white")
        ax.bar(x_pos, tn_pct, bottom=tp_pct + fn_pct + fp_pct, label="True Negative (TN)", color="#38a169", edgecolor="white")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{target.capitalize()} (tau = {TARGET_CONFIGS[target]['threshold']})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Child Age Group", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Proportion of Subgroup (%)", fontsize=11, fontweight="bold")
            ax.legend(loc="lower left", fontsize=9, frameon=True)

    plt.suptitle("Classification Composition by Child Age Group (Validation Cohort)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig38_path = os.path.join(FIGURES_DIR, "38_error_distribution_by_age.png")
    plt.savefig(fig38_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Fig 38: {fig38_path}")


def analyze_cross_target_error_overlap(validation_preds):
    """
    Analyzes pairwise and 3-way overlap among missed positive cases (false negatives)
    for children who are eligible across multiple outcomes in the validation set.
    """
    print("\n[*] Analyzing cross-target error overlap on validation set...")

    # Filter to children jointly eligible for all 3 targets
    joint_mask = (
        validation_preds["eligible_stunting"] &
        validation_preds["eligible_underweight"] &
        validation_preds["eligible_wasting"]
    )
    df_joint = validation_preds[joint_mask].copy()
    n_joint = len(df_joint)
    print(f"  Children eligible across all 3 targets: {n_joint:,}")

    # Determine FN status for each target
    # FN = actual positive (y=1) AND predicted negative (pred=0)
    for t in ["stunting", "underweight", "wasting"]:
        df_joint[f"fn_{t}"] = (df_joint[f"actual_{t}"] == 1) & (df_joint[f"pred_{t}"] == 0)
        df_joint[f"pos_{t}"] = (df_joint[f"actual_{t}"] == 1)

    fn_stunt = int(df_joint["fn_stunting"].sum())
    fn_under = int(df_joint["fn_underweight"].sum())
    fn_waste = int(df_joint["fn_wasting"].sum())

    pos_stunt = int(df_joint["pos_stunting"].sum())
    pos_under = int(df_joint["pos_underweight"].sum())
    pos_waste = int(df_joint["pos_wasting"].sum())

    # Pairwise FN overlap among co-occurring conditions
    # e.g., child is positive for both stunting & underweight, and missed on both
    co_pos_stunt_under = (df_joint["pos_stunting"] & df_joint["pos_underweight"])
    overlap_fn_stunt_under = int((df_joint["fn_stunting"] & df_joint["fn_underweight"]).sum())

    co_pos_stunt_waste = (df_joint["pos_stunting"] & df_joint["pos_wasting"])
    overlap_fn_stunt_waste = int((df_joint["fn_stunting"] & df_joint["fn_wasting"]).sum())

    co_pos_under_waste = (df_joint["pos_underweight"] & df_joint["pos_wasting"])
    overlap_fn_under_waste = int((df_joint["fn_underweight"] & df_joint["fn_wasting"]).sum())

    # Triple condition co-occurrence and triple FN
    co_pos_all_three = (df_joint["pos_stunting"] & df_joint["pos_underweight"] & df_joint["pos_wasting"])
    overlap_fn_all_three = int((df_joint["fn_stunting"] & df_joint["fn_underweight"] & df_joint["fn_wasting"]).sum())

    overlap_results = {
        "description": "Descriptive overlap of classification false negatives across malnutrition outcomes",
        "sample_size_joint_eligible": n_joint,
        "cohort": "validation",
        "single_target_fn_counts": {
            "stunting": {"actual_positives": pos_stunt, "false_negatives": fn_stunt, "fnr": round(fn_stunt / pos_stunt, 4) if pos_stunt > 0 else 0},
            "underweight": {"actual_positives": pos_under, "false_negatives": fn_under, "fnr": round(fn_under / pos_under, 4) if pos_under > 0 else 0},
            "wasting": {"actual_positives": pos_waste, "false_negatives": fn_waste, "fnr": round(fn_waste / pos_waste, 4) if pos_waste > 0 else 0}
        },
        "pairwise_overlap": {
            "stunting_and_underweight": {
                "jointly_positive_children": int(co_pos_stunt_under.sum()),
                "joint_false_negatives": overlap_fn_stunt_under,
                "overlap_proportion_of_joint_positives": round(overlap_fn_stunt_under / int(co_pos_stunt_under.sum()), 4) if int(co_pos_stunt_under.sum()) > 0 else 0
            },
            "stunting_and_wasting": {
                "jointly_positive_children": int(co_pos_stunt_waste.sum()),
                "joint_false_negatives": overlap_fn_stunt_waste,
                "overlap_proportion_of_joint_positives": round(overlap_fn_stunt_waste / int(co_pos_stunt_waste.sum()), 4) if int(co_pos_stunt_waste.sum()) > 0 else 0
            },
            "underweight_and_wasting": {
                "jointly_positive_children": int(co_pos_under_waste.sum()),
                "joint_false_negatives": overlap_fn_under_waste,
                "overlap_proportion_of_joint_positives": round(overlap_fn_under_waste / int(co_pos_under_waste.sum()), 4) if int(co_pos_under_waste.sum()) > 0 else 0
            }
        },
        "triple_overlap": {
            "triple_positive_children": int(co_pos_all_three.sum()),
            "triple_false_negatives": overlap_fn_all_three,
            "triple_fn_proportion": round(overlap_fn_all_three / int(co_pos_all_three.sum()), 4) if int(co_pos_all_three.sum()) > 0 else 0
        },
        "interpretation_note": "Overlap numbers describe empirical joint misses within the validation sample. They describe model behavior and do not establish a shared biological or clinical mechanism."
    }

    with open(OUTPUT_OVERLAP_JSON, "w") as f:
        json.dump(overlap_results, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved Error Overlap JSON: {OUTPUT_OVERLAP_JSON}")

    return overlap_results


def run_error_analysis():
    """Main execution orchestrator for Step 15 Error Analysis & Subgroup Audit."""
    start_time = time.time()
    print("=" * 80)
    print("STEP 15: DESCRIPTIVE ERROR ANALYSIS & SUBGROUP AUDIT (VALIDATION COHORT)")
    print("=" * 80)

    # 1. Verify Raw Dataset Immutability
    verify_raw_dataset_immutability()

    # 2. Load Processed Dataset
    print(f"\n[*] Loading processed dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    val_mask = df_raw["split"] == "val"
    df_val = df_raw[val_mask].copy()
    n_val_total = len(df_val)
    print(f"    Loaded {len(df_raw):,} records. Validation cohort size: {n_val_total:,}")

    # 3. Transform candidate features (34 approved predictors)
    print("[*] Generating Scenario A candidate feature matrix for validation cohort...")
    X_val = transform_candidate_features(df_val)
    print(f"[OK] Candidate feature matrix constructed: {X_val.shape[0]:,} rows x {X_val.shape[1]} features.")

    # 4. Generate validation predictions for all three targets
    print("\n[*] Generating validation predictions using locked model checkpoints and thresholds...")
    validation_preds = df_val[["eligible_stunting", "eligible_underweight", "eligible_wasting"]].copy()

    for target, config in TARGET_CONFIGS.items():
        model_file = config["model_file"]
        threshold = config["threshold"]
        elig_col = config["eligibility_col"]

        pipeline = joblib.load(model_file)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        y_true = df_val[target].fillna(-1).astype(int).values

        validation_preds[f"prob_{target}"] = y_prob
        validation_preds[f"pred_{target}"] = y_pred
        validation_preds[f"actual_{target}"] = y_true

        # Overall validation performance check
        t_mask = validation_preds[elig_col] == True
        overall = compute_contingency_metrics(y_true[t_mask], y_pred[t_mask])
        print(f"  {target.upper():12s} (tau={threshold:.2f}): N={overall['sample_size']:,}, "
              f"Sens={overall['sensitivity']*100:.2f}%, Spec={overall['specificity']*100:.2f}%, "
              f"PPV={overall['ppv']*100:.2f}%, F1={overall['f1_score']:.4f}, "
              f"TP={overall['tp']:,}, FP={overall['fp']:,}, TN={overall['tn']:,}, FN={overall['fn']:,}")

    # Attach subgroup features to validation_preds
    for dim in SUBGROUP_DIMENSIONS:
        feat = dim["feature"]
        validation_preds[feat] = X_val[feat].values

    # 5. Execute Subgroup Error Evaluations
    print("\n[*] Calculating subgroup contingency metrics across all dimensions...")
    all_metrics_rows = []
    nested_json_results = {
        "step": "Step 15 — Descriptive Error Analysis & Subgroup Analysis",
        "cohort": "Validation cohort (split == 'val')",
        "validation_total_records": n_val_total,
        "minimum_sample_size_rule": MIN_SUBGROUP_N,
        "locked_targets": {t: {"threshold": c["threshold"], "model": c["model_id"]} for t, c in TARGET_CONFIGS.items()},
        "subgroup_results": {}
    }

    for target, config in TARGET_CONFIGS.items():
        elig_col = config["eligibility_col"]
        threshold = config["threshold"]
        nested_json_results["subgroup_results"][target] = {}

        t_mask = validation_preds[elig_col] == True
        df_target_val = validation_preds[t_mask]

        for dim in SUBGROUP_DIMENSIONS:
            dim_name = dim["name"]
            feat = dim["feature"]
            disp_name = dim["display_name"]
            labels_map = dim["labels"]
            order = dim["order"]

            nested_json_results["subgroup_results"][target][dim_name] = []

            for cat_val in order:
                cat_mask = (df_target_val[feat] == cat_val)
                sub_df = df_target_val[cat_mask]
                cat_label = labels_map.get(cat_val, str(cat_val))
                sub_n = len(sub_df)

                if sub_n == 0:
                    continue

                y_true_sub = sub_df[f"actual_{target}"].values
                y_pred_sub = sub_df[f"pred_{target}"].values

                metrics = compute_contingency_metrics(y_true_sub, y_pred_sub)

                # Compute bootstrap CIs if subgroup meets minimum sample rule
                is_evaluable = sub_n >= MIN_SUBGROUP_N
                if is_evaluable and metrics["actual_positives"] >= 10 and metrics["actual_negatives"] >= 10:
                    cis = compute_bootstrap_cis(y_true_sub, y_pred_sub, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
                else:
                    cis = {
                        "sensitivity_95ci": {"ci_lower": None, "ci_upper": None},
                        "specificity_95ci": {"ci_lower": None, "ci_upper": None},
                        "ppv_95ci": {"ci_lower": None, "ci_upper": None},
                        "f1_score_95ci": {"ci_lower": None, "ci_upper": None}
                    }

                record = {
                    "target": target,
                    "threshold": threshold,
                    "dimension": dim_name,
                    "dimension_display": disp_name,
                    "category_raw": str(cat_val),
                    "category_label": cat_label,
                    "sample_size": metrics["sample_size"],
                    "is_comparative_evaluable": is_evaluable,
                    "actual_positives": metrics["actual_positives"],
                    "actual_negatives": metrics["actual_negatives"],
                    "prevalence": round(metrics["prevalence"], 4),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"],
                    "sensitivity": round(metrics["sensitivity"], 4) if not np.isnan(metrics["sensitivity"]) else None,
                    "sensitivity_ci_lower": cis["sensitivity_95ci"]["ci_lower"],
                    "sensitivity_ci_upper": cis["sensitivity_95ci"]["ci_upper"],
                    "specificity": round(metrics["specificity"], 4) if not np.isnan(metrics["specificity"]) else None,
                    "specificity_ci_lower": cis["specificity_95ci"]["ci_lower"],
                    "specificity_ci_upper": cis["specificity_95ci"]["ci_upper"],
                    "false_negative_rate": round(metrics["false_negative_rate"], 4) if not np.isnan(metrics["false_negative_rate"]) else None,
                    "false_positive_rate": round(metrics["false_positive_rate"], 4) if not np.isnan(metrics["false_positive_rate"]) else None,
                    "ppv": round(metrics["ppv"], 4) if not np.isnan(metrics["ppv"]) else None,
                    "ppv_ci_lower": cis["ppv_95ci"]["ci_lower"],
                    "ppv_ci_upper": cis["ppv_95ci"]["ci_upper"],
                    "npv": round(metrics["npv"], 4) if not np.isnan(metrics["npv"]) else None,
                    "f1_score": round(metrics["f1_score"], 4) if not np.isnan(metrics["f1_score"]) else None,
                    "f1_score_ci_lower": cis["f1_score_95ci"]["ci_lower"],
                    "f1_score_ci_upper": cis["f1_score_95ci"]["ci_upper"]
                }
                all_metrics_rows.append(record)
                nested_json_results["subgroup_results"][target][dim_name].append(record)

    # 6. Save Metrics CSV and JSON
    df_metrics = pd.DataFrame(all_metrics_rows)
    df_metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n[+] Saved Subgroup Error Metrics CSV: {OUTPUT_METRICS_CSV} ({len(df_metrics)} rows)")

    with open(OUTPUT_METRICS_JSON, "w") as f:
        json.dump(nested_json_results, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved Subgroup Error Metrics JSON: {OUTPUT_METRICS_JSON}")

    # 7. Analyze Cross-Target Error Overlap
    analyze_cross_target_error_overlap(validation_preds)

    # 8. Generate Visualizations (Figures 33 to 38)
    print("\n[*] Generating publication-quality error analysis figures...")
    generate_error_visualizations(df_metrics)

    elapsed = time.time() - start_time
    print(f"\n[OK] Step 15 Error Analysis executed successfully in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    run_error_analysis()
