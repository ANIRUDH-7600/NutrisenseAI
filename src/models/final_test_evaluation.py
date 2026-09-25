#!/usr/bin/env python3
"""
Step 13: Final Locked Test Evaluation of Pre-Selected LightGBM Configurations
Project: NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence
Dataset: NFHS-5 India 2019–21 Children's Recode (KR)
Setting: Scenario A — Community Pre-Screening (34 non-invasive features)

Evaluates the champion LightGBM Unweighted models on the held-out locked test partition (N = 33,069).
STRICT RULES:
1. Zero model retraining — loads existing fitted pipelines.
2. Zero threshold optimization — applies exact validation-derived thresholds (0.35, 0.31, 0.17).
3. Zero post-hoc calibration fitting on test labels.
4. Bootstrap 95% CIs evaluated strictly within test set on fixed predictions.
5. Absolute privacy: no individual microdata records are exported.
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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix
)

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.features.build_features import transform_candidate_features, transform_candidate_features_v2

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745


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

DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
VALIDATION_METRICS_PATH = "data/interim/threshold_analysis_metrics.json"

FEATURE_METADATA_PATH_V2 = "data/interim/feature_metadata_v2.json"
VALIDATION_METRICS_PATH_V2 = "data/interim/threshold_analysis_metrics_v2.json"

MODELS_DIR = "models"
MODELS_DIR_V2 = "models/v2"
FIGURES_DIR = "reports/figures"

OUTPUT_TEST_JSON = "data/interim/final_test_metrics.json"
OUTPUT_TEST_CSV = "data/interim/final_test_metrics.csv"
OUTPUT_VAL_VS_TEST_CSV = "data/interim/validation_vs_test_metrics.csv"

OUTPUT_TEST_JSON_V2 = "data/interim/final_test_metrics_v2.json"
OUTPUT_TEST_CSV_V2 = "data/interim/final_test_metrics_v2.csv"
OUTPUT_VAL_VS_TEST_CSV_V2 = "data/interim/validation_vs_test_metrics_v2.csv"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000

LOCKED_CONFIGURATIONS = {
    "stunting": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_stunting_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.35
    },
    "underweight": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_underweight_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.31
    },
    "wasting": {
        "model_file": os.path.join(MODELS_DIR, "model_comparison_lightgbm_wasting_unweighted.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.17
    }
}

LOCKED_CONFIGURATIONS_V2 = {
    "stunting": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_stunting_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.36
    },
    "underweight": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_underweight_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.30
    },
    "wasting": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_wasting_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.17
    }
}


def verify_raw_dataset_immutability():
    """Verify raw DHS data byte size integrity."""
    if not os.path.exists(RAW_DTA_PATH):
        raise FileNotFoundError(f"Raw dataset {RAW_DTA_PATH} not found.")
    actual_bytes = os.path.getsize(RAW_DTA_PATH)
    if actual_bytes != EXPECTED_RAW_BYTES:
        raise ValueError(f"Raw dataset altered! Expected {EXPECTED_RAW_BYTES}, got {actual_bytes}.")
    print(f"[OK] Raw dataset immutability verified ({actual_bytes:,} bytes).")


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    """Computes full diagnostic metrics at a fixed decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    n = len(y_true)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    f1 = (2.0 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
    youden_j = sens + spec - 1.0

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "sample_size": n,
        "actual_positives": tp + fn,
        "actual_negatives": tn + fp,
        "predicted_positives": tp + fp,
        "predicted_negatives": tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unweighted_test_prevalence": (tp + fn) / n if n > 0 else 0.0,
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "precision_ppv": ppv,
        "npv": npv,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "youden_j": youden_j,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier
    }


def compute_bootstrap_confidence_intervals(y_true, y_prob, threshold, n_boot=1000, seed=42):
    """
    Computes 95% non-parametric bootstrap confidence intervals on fixed predictions within the test set.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "sensitivity": [],
        "specificity": [],
        "precision_ppv": [],
        "f1_score": [],
        "balanced_accuracy": []
    }

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)

    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        b_y = y_true_arr[indices]
        b_p = y_prob_arr[indices]

        # Guard against zero positive or zero negative in rare bootstrap resample
        if len(np.unique(b_y)) < 2:
            continue

        b_pred = (b_p >= threshold).astype(int)
        tp = np.sum((b_y == 1) & (b_pred == 1))
        fp = np.sum((b_y == 0) & (b_pred == 1))
        tn = np.sum((b_y == 0) & (b_pred == 0))
        fn = np.sum((b_y == 1) & (b_pred == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0

        roc_auc = roc_auc_score(b_y, b_p)
        pr_auc = average_precision_score(b_y, b_p)

        boot_metrics["roc_auc"].append(roc_auc)
        boot_metrics["pr_auc"].append(pr_auc)
        boot_metrics["sensitivity"].append(sens)
        boot_metrics["specificity"].append(spec)
        boot_metrics["precision_ppv"].append(ppv)
        boot_metrics["f1_score"].append(f1)
        boot_metrics["balanced_accuracy"].append(bal_acc)

    ci_results = {}
    for metric_name, values in boot_metrics.items():
        if len(values) > 0:
            low = float(np.percentile(values, 2.5))
            high = float(np.percentile(values, 97.5))
            ci_results[metric_name] = {
                "ci_lower_95": round(low, 4),
                "ci_upper_95": round(high, 4)
            }
        else:
            ci_results[metric_name] = {"ci_lower_95": None, "ci_upper_95": None}

    return ci_results


def plot_test_confusion_matrices(test_results, suffix=""):
    """Plots normalized and raw 2x2 confusion matrices for all 3 targets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    targets = ["stunting", "underweight", "wasting"]

    for idx, target in enumerate(targets):
        res = test_results[target]["metrics"]
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        annot = np.array([
            [f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)"],
            [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)"]
        ])

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 11, "weight": "bold"}
        )
        thresh = test_results[target]["locked_threshold"]
        axes[idx].set_title(f"{target.capitalize()} (Locked $\\tau = {thresh:.2f}$)\nTest $N = {res['sample_size']:,}$", fontsize=12, pad=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Ground Truth", fontsize=10)
        axes[idx].set_xticklabels(["Eunourished (0)", "Malnourished (1)"])
        axes[idx].set_yticklabels(["Eunourished (0)", "Malnourished (1)"])

    title_suffix = " (30-Feature v2)" if suffix == "_v2" else ""
    plt.suptitle(f"Figure 19: Locked Test-Set Contingency Matrices (Scenario A Community Pre-Screening{title_suffix})", fontsize=14, weight="bold", y=1.03)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"fig19_test_confusion_matrices{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 19 to: {out_path}")


def plot_validation_vs_test_bars(val_vs_test_df, suffix=""):
    """Plots comparative validation vs test metrics across targets."""
    metrics_to_plot = ["roc_auc", "pr_auc", "sensitivity", "specificity", "f1_score"]
    metric_labels = ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "F1-Score"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    targets = ["stunting", "underweight", "wasting"]

    for idx, target in enumerate(targets):
        t_df = val_vs_test_df[val_vs_test_df["target"] == target].iloc[0]
        val_vals = [t_df[f"val_{m}"] for m in metrics_to_plot]
        test_vals = [t_df[f"test_{m}"] for m in metrics_to_plot]

        x = np.arange(len(metrics_to_plot))
        width = 0.35

        axes[idx].bar(x - width/2, val_vals, width, label="Validation", color="#2b5c8f", alpha=0.9)
        axes[idx].bar(x + width/2, test_vals, width, label="Locked Test", color="#e27c38", alpha=0.9)

        axes[idx].set_title(f"{target.capitalize()} (Threshold $\\tau = {t_df['threshold']:.2f}$)", fontsize=12, pad=10)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=9)
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            axes[idx].set_ylabel("Metric Value", fontsize=11)
            axes[idx].legend(loc="upper right")

    title_suffix = " (30-Feature v2)" if suffix == "_v2" else ""
    plt.suptitle(f"Figure 20: Validation vs. Locked Test Metric Comparison Across Targets{title_suffix}", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"fig20_validation_vs_test_metrics{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Figure 20 to: {out_path}")


def plot_test_roc_and_pr_curves(test_curve_data, suffix=""):
    """Plots Test ROC Curves (Fig 21) and Test PR Curves (Fig 22)."""
    # 1. ROC Curves
    plt.figure(figsize=(7, 6))
    for target, c_data in test_curve_data.items():
        fpr = c_data["fpr"]
        tpr = c_data["tpr"]
        auc = c_data["roc_auc"]
        thresh = c_data["threshold"]
        op_fpr = c_data["op_fpr"]
        op_tpr = c_data["op_tpr"]
        plt.plot(fpr, tpr, lw=2, label=f"{target.capitalize()} (AUC = {auc:.4f})")
        plt.scatter([op_fpr], [op_tpr], s=80, zorder=5, edgecolors="black", label=f"  $\\tau = {thresh:.2f}$ Operating Point")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Chance Diagonal (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    title_suffix = " (30-Feature v2)" if suffix == "_v2" else ""
    plt.title(f"Figure 21: Test-Set ROC Curves (Locked Pre-Screening LightGBM{title_suffix})", fontsize=12, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path_roc = os.path.join(FIGURES_DIR, f"fig21_test_roc_curves{suffix}.png")
    plt.savefig(out_path_roc, dpi=300)
    plt.close()
    print(f"[+] Saved Figure 21 to: {out_path_roc}")

    # 2. PR Curves
    plt.figure(figsize=(7, 6))
    for target, c_data in test_curve_data.items():
        rec = c_data["pr_recall"]
        prec = c_data["pr_precision"]
        pr_auc = c_data["pr_auc"]
        prev = c_data["prevalence"]
        thresh = c_data["threshold"]
        op_rec = c_data["op_sensitivity"]
        op_prec = c_data["op_precision"]
        line = plt.plot(rec, prec, lw=2, label=f"{target.capitalize()} (PR-AUC = {pr_auc:.4f})")
        plt.axhline(prev, color=line[0].get_color(), linestyle=":", alpha=0.6, label=f"  {target.capitalize()} Base Rate ({prev*100:.1f}%)")
        plt.scatter([op_rec], [op_prec], s=80, zorder=5, edgecolors="black")

    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (PPV)", fontsize=11)
    plt.title(f"Figure 22: Test-Set Precision-Recall Curves with Locked Operating Points{title_suffix}", fontsize=12, weight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path_pr = os.path.join(FIGURES_DIR, f"fig22_test_precision_recall_curves{suffix}.png")
    plt.savefig(out_path_pr, dpi=300)
    plt.close()
    print(f"[+] Saved Figure 22 to: {out_path_pr}")


def run_final_test_evaluation():
    print("================================================================================")
    print("NutriSense AI — Step 13: Final Locked Test Evaluation")
    print("================================================================================")
    print("Dataset: NFHS-5 India 2019-21 Children's Recode (KR)")
    print("Locked Test Partition: N = 33,069 children strictly")
    print("STRICT RULE: Zero retraining, zero threshold re-optimization.\n")

    # 1. Data Integrity Check
    verify_raw_dataset_immutability()

    # 2. Load Processed Dataset and Feature Matrix
    print(f"\n[*] Loading processed dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"    Loaded {len(df_raw):,} records. Partitions: {df_raw['split'].value_counts().to_dict()}")

    test_total = (df_raw["split"] == "test").sum()
    assert test_total == 33069, f"Unexpected test partition size! Expected 33,069, found {test_total}."
    print(f"[OK] Test partition confirmed at exactly {test_total:,} children.")

    # 3. Transform candidate features (34 approved predictors)
    print("[*] Generating Scenario A candidate feature matrix X (34 non-invasive features)...")
    X_all = transform_candidate_features(df_raw)
    print(f"[OK] Feature matrix constructed: {X_all.shape[0]:,} rows x {X_all.shape[1]} features.")

    # 4. Load Step 11 validation metrics for comparative table
    with open(VALIDATION_METRICS_PATH, "r") as f:
        val_data = json.load(f)["threshold_summary"]

    test_results = {}
    val_vs_test_rows = []
    test_metrics_rows = []
    test_curve_data = {}

    for target, config in LOCKED_CONFIGURATIONS.items():
        print("\n" + "=" * 75)
        print(f"EVALUATING LOCKED TEST SET — TARGET: {target.upper()}")
        print("=" * 75)

        model_path = config["model_file"]
        model_id = config["model_id"]
        threshold = config["threshold"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing champion model artifact: {model_path}")

        print(f"  Model Artifact: {model_path}")
        print(f"  Locked Decision Threshold: tau = {threshold:.2f}")

        # Load fitted pipeline
        pipeline = joblib.load(model_path)
        print(f"  Loaded Pipeline: Steps = {[s[0] for s in pipeline.steps]}")

        # Filter test set by target anthropometric eligibility
        elig_col = f"eligible_{target}"
        test_mask = (df_raw["split"] == "test") & (df_raw[elig_col] == True)
        X_test = X_all.loc[test_mask].copy()
        y_test = df_raw.loc[test_mask, target].astype(int).values

        n_test = len(X_test)
        pos_test = int(np.sum(y_test == 1))
        neg_test = int(np.sum(y_test == 0))
        prev_test = pos_test / n_test

        print(f"  Evaluated Test Cohort: N = {n_test:,} (Pos: {pos_test:,}, Neg: {neg_test:,}, Prev: {prev_test*100:.2f}%)")

        # Generate predictions strictly on fixed pipeline
        t0 = time.time()
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t0
        latency_per_sample_ms = (inference_time / n_test) * 1000.0
        print(f"  Inference Completed: {inference_time:.3f} sec ({latency_per_sample_ms:.3f} ms/sample)")

        # Compute point metrics at locked threshold
        metrics = compute_metrics_at_threshold(y_test, y_prob, threshold)

        # Compute bootstrap 95% confidence intervals
        print(f"  Computing {N_BOOTSTRAP} bootstrap 95% confidence intervals within test set...")
        t_boot = time.time()
        cis = compute_bootstrap_confidence_intervals(y_test, y_prob, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        print(f"  Bootstrap completed in {time.time() - t_boot:.2f} sec.")

        # Store curve data for figures
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        pr_prec_arr, pr_rec_arr, _ = precision_recall_curve(y_test, y_prob)

        test_curve_data[target] = {
            "fpr": fpr_arr,
            "tpr": tpr_arr,
            "roc_auc": metrics["roc_auc"],
            "threshold": threshold,
            "op_fpr": metrics["fp"] / metrics["actual_negatives"],
            "op_tpr": metrics["sensitivity"],
            "pr_precision": pr_prec_arr,
            "pr_recall": pr_rec_arr,
            "pr_auc": metrics["pr_auc"],
            "prevalence": prev_test,
            "op_sensitivity": metrics["sensitivity"],
            "op_precision": metrics["precision_ppv"]
        }

        # Retrieve validation metrics from Step 11 for comparison
        # LightGBM (Unweighted) at specified threshold
        val_t_models = val_data[target]["models"]["LightGBM (Unweighted)"]
        val_m_info = val_t_models["max_youden_j_threshold"]

        val_roc = val_t_models["roc_auc"]
        val_pr = val_t_models["pr_auc"]
        val_sens = val_m_info["recall_sensitivity"]
        val_spec = val_m_info["specificity"]
        val_f1 = val_m_info["f1_score"]
        val_bal_acc = val_m_info["balanced_accuracy"]
        val_ppv = val_m_info["precision"]

        # Build comparison row
        comp_row = {
            "target": target,
            "model": model_id,
            "threshold": threshold,
            "val_roc_auc": round(val_roc, 4),
            "test_roc_auc": round(metrics["roc_auc"], 4),
            "delta_roc_auc": round(metrics["roc_auc"] - val_roc, 4),
            "val_pr_auc": round(val_pr, 4),
            "test_pr_auc": round(metrics["pr_auc"], 4),
            "delta_pr_auc": round(metrics["pr_auc"] - val_pr, 4),
            "val_sensitivity": round(val_sens, 4),
            "test_sensitivity": round(metrics["sensitivity"], 4),
            "delta_sensitivity": round(metrics["sensitivity"] - val_sens, 4),
            "val_specificity": round(val_spec, 4),
            "test_specificity": round(metrics["specificity"], 4),
            "delta_specificity": round(metrics["specificity"] - val_spec, 4),
            "val_f1_score": round(val_f1, 4),
            "test_f1_score": round(metrics["f1_score"], 4),
            "delta_f1_score": round(metrics["f1_score"] - val_f1, 4),
            "val_balanced_acc": round(val_bal_acc, 4),
            "test_balanced_acc": round(metrics["balanced_accuracy"], 4),
            "delta_balanced_acc": round(metrics["balanced_accuracy"] - val_bal_acc, 4)
        }
        val_vs_test_rows.append(comp_row)

        test_result_record = {
            "target": target,
            "model_id": model_id,
            "locked_threshold": threshold,
            "metrics": metrics,
            "confidence_intervals_95": cis,
            "validation_comparison": {
                "val_roc_auc": val_roc,
                "val_pr_auc": val_pr,
                "val_sensitivity": val_sens,
                "val_specificity": val_spec,
                "val_f1": val_f1,
                "val_balanced_acc": val_bal_acc
            }
        }
        test_results[target] = test_result_record

        # CSV row
        test_csv_row = {
            "target": target,
            "model": model_id,
            "threshold": threshold,
            "sample_size": n_test,
            "unweighted_prevalence": round(prev_test, 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "roc_auc_ci_lower": cis["roc_auc"]["ci_lower_95"],
            "roc_auc_ci_upper": cis["roc_auc"]["ci_upper_95"],
            "pr_auc": round(metrics["pr_auc"], 4),
            "pr_auc_ci_lower": cis["pr_auc"]["ci_lower_95"],
            "pr_auc_ci_upper": cis["pr_auc"]["ci_upper_95"],
            "sensitivity": round(metrics["sensitivity"], 4),
            "sensitivity_ci_lower": cis["sensitivity"]["ci_lower_95"],
            "sensitivity_ci_upper": cis["sensitivity"]["ci_upper_95"],
            "specificity": round(metrics["specificity"], 4),
            "specificity_ci_lower": cis["specificity"]["ci_lower_95"],
            "specificity_ci_upper": cis["specificity"]["ci_upper_95"],
            "precision_ppv": round(metrics["precision_ppv"], 4),
            "precision_ci_lower": cis["precision_ppv"]["ci_lower_95"],
            "precision_ci_upper": cis["precision_ppv"]["ci_upper_95"],
            "f1_score": round(metrics["f1_score"], 4),
            "f1_ci_lower": cis["f1_score"]["ci_lower_95"],
            "f1_ci_upper": cis["f1_score"]["ci_upper_95"],
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "youden_j": round(metrics["youden_j"], 4),
            "brier_score": round(metrics["brier_score"], 4),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"]
        }
        test_metrics_rows.append(test_csv_row)

        print(f"  TEST METRICS: ROC-AUC={metrics['roc_auc']:.4f} [95% CI: {cis['roc_auc']['ci_lower_95']}-{cis['roc_auc']['ci_upper_95']}], PR-AUC={metrics['pr_auc']:.4f}")
        print(f"  SCREENING PERFORMANCE: Sens={metrics['sensitivity']*100:.2f}%, Spec={metrics['specificity']*100:.2f}%, PPV={metrics['precision_ppv']*100:.2f}%, F1={metrics['f1_score']:.4f}, BalAcc={metrics['balanced_accuracy']*100:.2f}%")
        print(f"  CONFUSION MATRIX: TP={metrics['tp']:,}, FP={metrics['fp']:,}, TN={metrics['tn']:,}, FN={metrics['fn']:,}")
        print(f"  DELTA VS VAL: dROC={comp_row['delta_roc_auc']:+.4f}, dSens={comp_row['delta_sensitivity']*100:+.2f}%, dSpec={comp_row['delta_specificity']*100:+.2f}%, dF1={comp_row['delta_f1_score']:+.4f}")

    # 5. Save Data Artifacts
    val_vs_test_df = pd.DataFrame(val_vs_test_rows)
    val_vs_test_df.to_csv(OUTPUT_VAL_VS_TEST_CSV, index=False)
    print(f"\n[+] Saved validation vs test comparison table to: {OUTPUT_VAL_VS_TEST_CSV}")

    test_metrics_df = pd.DataFrame(test_metrics_rows)
    test_metrics_df.to_csv(OUTPUT_TEST_CSV, index=False)
    print(f"[+] Saved test metrics CSV to: {OUTPUT_TEST_CSV}")

    final_json_payload = {
        "step": "Step 13 — Final Locked Test Evaluation",
        "scenario": "Scenario A — Community Pre-Screening",
        "dataset": "NFHS-5 India 2019-21 Children's Recode (KR)",
        "test_partition_total_records": test_total,
        "evaluation_principle": "Strictly locked out-of-sample evaluation with zero retraining and zero threshold re-tuning",
        "bootstrap_iterations": N_BOOTSTRAP,
        "test_results": test_results
    }

    with open(OUTPUT_TEST_JSON, "w") as f:
        json.dump(final_json_payload, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved structured test metrics JSON to: {OUTPUT_TEST_JSON}")

    # 6. Generate Figures
    print("\n[*] Generating publication-quality figures for Step 13...")
    plot_test_confusion_matrices(test_results)
    plot_validation_vs_test_bars(val_vs_test_df)
    plot_test_roc_and_pr_curves(test_curve_data)

    print("\n================================================================================")
    print("STEP 13 FINAL LOCKED TEST EVALUATION COMPLETED SUCCESSFULLY!")
    print("================================================================================")


def run_final_test_evaluation_v2():
    print("================================================================================")
    print("NutriSense AI — Major Step 5: Final Locked Test Evaluation (30 Predictors v2)")
    print("================================================================================")
    print("Dataset: NFHS-5 India 2019-21 Children's Recode (KR)")
    print("Locked Test Partition: N = 33,069 children strictly")
    print("STRICT RULE: Zero retraining, zero threshold re-optimization.\n")

    # 1. Data Integrity Check
    verify_raw_dataset_immutability()

    # 2. Load Processed Dataset and Feature Matrix
    print(f"\n[*] Loading processed dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"    Loaded {len(df_raw):,} records. Partitions: {df_raw['split'].value_counts().to_dict()}")

    test_total = (df_raw["split"] == "test").sum()
    assert test_total == 33069, f"Unexpected test partition size! Expected 33,069, found {test_total}."
    print(f"[OK] Test partition confirmed at exactly {test_total:,} children.")

    # 3. Transform candidate features (30 approved predictors for v2)
    print("[*] Generating Scenario A v2 candidate feature matrix X (30 non-invasive features)...")
    X_all = transform_candidate_features_v2(df_raw)
    print(f"[OK] Feature matrix constructed: {X_all.shape[0]:,} rows x {X_all.shape[1]} features.")
    assert X_all.shape[1] == 30, f"Expected 30 features, got {X_all.shape[1]}"

    # 4. Load Step 4 validation metrics for comparative table
    with open(VALIDATION_METRICS_PATH_V2, "r") as f:
        val_data = json.load(f)["threshold_summary"]

    test_results = {}
    val_vs_test_rows = []
    test_metrics_rows = []
    test_curve_data = {}

    for target, config in LOCKED_CONFIGURATIONS_V2.items():
        print("\n" + "=" * 75)
        print(f"EVALUATING LOCKED TEST SET (v2) — TARGET: {target.upper()}")
        print("=" * 75)

        model_path = config["model_file"]
        model_id = config["model_id"]
        threshold = config["threshold"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing champion model artifact: {model_path}")

        print(f"  Model Artifact: {model_path}")
        print(f"  Locked Decision Threshold: tau = {threshold:.2f}")

        # Load fitted pipeline
        pipeline = joblib.load(model_path)
        print(f"  Loaded Pipeline: Steps = {[s[0] for s in pipeline.steps]}")

        # Filter test set by target anthropometric eligibility
        elig_col = f"eligible_{target}"
        test_mask = (df_raw["split"] == "test") & (df_raw[elig_col] == True)
        X_test = X_all.loc[test_mask].copy()
        y_test = df_raw.loc[test_mask, target].astype(int).values

        n_test = len(X_test)
        pos_test = int(np.sum(y_test == 1))
        neg_test = int(np.sum(y_test == 0))
        prev_test = pos_test / n_test

        print(f"  Evaluated Test Cohort: N = {n_test:,} (Pos: {pos_test:,}, Neg: {neg_test:,}, Prev: {prev_test*100:.2f}%)")

        # Generate predictions strictly on fixed pipeline
        t0 = time.time()
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t0
        latency_per_sample_ms = (inference_time / n_test) * 1000.0
        print(f"  Inference Completed: {inference_time:.3f} sec ({latency_per_sample_ms:.3f} ms/sample)")

        # Compute point metrics at locked threshold
        metrics = compute_metrics_at_threshold(y_test, y_prob, threshold)

        # Compute bootstrap 95% confidence intervals
        print(f"  Computing {N_BOOTSTRAP} bootstrap 95% confidence intervals within test set...")
        t_boot = time.time()
        cis = compute_bootstrap_confidence_intervals(y_test, y_prob, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED)
        print(f"  Bootstrap completed in {time.time() - t_boot:.2f} sec.")

        # Store curve data for figures
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        pr_prec_arr, pr_rec_arr, _ = precision_recall_curve(y_test, y_prob)

        test_curve_data[target] = {
            "fpr": fpr_arr,
            "tpr": tpr_arr,
            "roc_auc": metrics["roc_auc"],
            "threshold": threshold,
            "op_fpr": metrics["fp"] / metrics["actual_negatives"],
            "op_tpr": metrics["sensitivity"],
            "pr_precision": pr_prec_arr,
            "pr_recall": pr_rec_arr,
            "pr_auc": metrics["pr_auc"],
            "prevalence": prev_test,
            "op_sensitivity": metrics["sensitivity"],
            "op_precision": metrics["precision_ppv"]
        }

        # Retrieve validation metrics from Step 4 for comparison
        val_t_models = val_data[target]["models"]["LightGBM (Unweighted)"]
        val_m_info = val_t_models.get("validation_operating_threshold", val_t_models.get("max_youden_j_threshold"))

        val_roc = val_t_models["roc_auc"]
        val_pr = val_t_models["pr_auc"]
        val_sens = val_m_info["recall_sensitivity"]
        val_spec = val_m_info["specificity"]
        val_f1 = val_m_info["f1_score"]
        val_bal_acc = val_m_info["balanced_accuracy"]
        val_ppv = val_m_info["precision"]

        # Build comparison row
        comp_row = {
            "target": target,
            "model": model_id,
            "threshold": threshold,
            "val_roc_auc": round(val_roc, 4),
            "test_roc_auc": round(metrics["roc_auc"], 4),
            "delta_roc_auc": round(metrics["roc_auc"] - val_roc, 4),
            "val_pr_auc": round(val_pr, 4),
            "test_pr_auc": round(metrics["pr_auc"], 4),
            "delta_pr_auc": round(metrics["pr_auc"] - val_pr, 4),
            "val_sensitivity": round(val_sens, 4),
            "test_sensitivity": round(metrics["sensitivity"], 4),
            "delta_sensitivity": round(metrics["sensitivity"] - val_sens, 4),
            "val_specificity": round(val_spec, 4),
            "test_specificity": round(metrics["specificity"], 4),
            "delta_specificity": round(metrics["specificity"] - val_spec, 4),
            "val_f1_score": round(val_f1, 4),
            "test_f1_score": round(metrics["f1_score"], 4),
            "delta_f1_score": round(metrics["f1_score"] - val_f1, 4),
            "val_balanced_acc": round(val_bal_acc, 4),
            "test_balanced_acc": round(metrics["balanced_accuracy"], 4),
            "delta_balanced_acc": round(metrics["balanced_accuracy"] - val_bal_acc, 4)
        }
        val_vs_test_rows.append(comp_row)

        test_result_record = {
            "target": target,
            "model_id": model_id,
            "locked_threshold": threshold,
            "metrics": metrics,
            "confidence_intervals_95": cis,
            "validation_comparison": {
                "val_roc_auc": val_roc,
                "val_pr_auc": val_pr,
                "val_sensitivity": val_sens,
                "val_specificity": val_spec,
                "val_f1": val_f1,
                "val_balanced_acc": val_bal_acc
            }
        }
        test_results[target] = test_result_record

        # CSV row
        test_csv_row = {
            "target": target,
            "model": model_id,
            "threshold": threshold,
            "sample_size": n_test,
            "unweighted_prevalence": round(prev_test, 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "roc_auc_ci_lower": cis["roc_auc"]["ci_lower_95"],
            "roc_auc_ci_upper": cis["roc_auc"]["ci_upper_95"],
            "pr_auc": round(metrics["pr_auc"], 4),
            "pr_auc_ci_lower": cis["pr_auc"]["ci_lower_95"],
            "pr_auc_ci_upper": cis["pr_auc"]["ci_upper_95"],
            "sensitivity": round(metrics["sensitivity"], 4),
            "sensitivity_ci_lower": cis["sensitivity"]["ci_lower_95"],
            "sensitivity_ci_upper": cis["sensitivity"]["ci_upper_95"],
            "specificity": round(metrics["specificity"], 4),
            "specificity_ci_lower": cis["specificity"]["ci_lower_95"],
            "specificity_ci_upper": cis["specificity"]["ci_upper_95"],
            "precision_ppv": round(metrics["precision_ppv"], 4),
            "precision_ci_lower": cis["precision_ppv"]["ci_lower_95"],
            "precision_ci_upper": cis["precision_ppv"]["ci_upper_95"],
            "f1_score": round(metrics["f1_score"], 4),
            "f1_ci_lower": cis["f1_score"]["ci_lower_95"],
            "f1_ci_upper": cis["f1_score"]["ci_upper_95"],
            "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
            "youden_j": round(metrics["youden_j"], 4),
            "brier_score": round(metrics["brier_score"], 4),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"]
        }
        test_metrics_rows.append(test_csv_row)

        print(f"  TEST METRICS (v2): ROC-AUC={metrics['roc_auc']:.4f} [95% CI: {cis['roc_auc']['ci_lower_95']}-{cis['roc_auc']['ci_upper_95']}], PR-AUC={metrics['pr_auc']:.4f}")
        print(f"  SCREENING PERFORMANCE: Sens={metrics['sensitivity']*100:.2f}%, Spec={metrics['specificity']*100:.2f}%, PPV={metrics['precision_ppv']*100:.2f}%, F1={metrics['f1_score']:.4f}, BalAcc={metrics['balanced_accuracy']*100:.2f}%")
        print(f"  CONFUSION MATRIX: TP={metrics['tp']:,}, FP={metrics['fp']:,}, TN={metrics['tn']:,}, FN={metrics['fn']:,}")
        print(f"  DELTA VS VAL: dROC={comp_row['delta_roc_auc']:+.4f}, dSens={comp_row['delta_sensitivity']*100:+.2f}%, dSpec={comp_row['delta_specificity']*100:+.2f}%, dF1={comp_row['delta_f1_score']:+.4f}")

    # 5. Save Data Artifacts
    val_vs_test_df = pd.DataFrame(val_vs_test_rows)
    val_vs_test_df.to_csv(OUTPUT_VAL_VS_TEST_CSV_V2, index=False)
    print(f"\n[+] Saved validation vs test comparison table to: {OUTPUT_VAL_VS_TEST_CSV_V2}")

    test_metrics_df = pd.DataFrame(test_metrics_rows)
    test_metrics_df.to_csv(OUTPUT_TEST_CSV_V2, index=False)
    print(f"[+] Saved test metrics CSV to: {OUTPUT_TEST_CSV_V2}")

    final_json_payload = {
        "step": "Major Step 5 — Final Locked Test Evaluation (30 Predictors v2)",
        "scenario": "Scenario A — Community Pre-Screening",
        "feature_count": 30,
        "feature_version": "v2.0.0",
        "dataset": "NFHS-5 India 2019-21 Children's Recode (KR)",
        "test_partition_total_records": test_total,
        "evaluation_principle": "Strictly locked out-of-sample evaluation with zero retraining and zero threshold re-tuning",
        "bootstrap_iterations": N_BOOTSTRAP,
        "test_results": test_results
    }

    with open(OUTPUT_TEST_JSON_V2, "w") as f:
        json.dump(final_json_payload, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved structured test metrics JSON to: {OUTPUT_TEST_JSON_V2}")

    # 6. Generate Figures
    print("\n[*] Generating publication-quality figures for Step 5 (v2)...")
    plot_test_confusion_matrices(test_results, suffix="_v2")
    plot_validation_vs_test_bars(val_vs_test_df, suffix="_v2")
    plot_test_roc_and_pr_curves(test_curve_data, suffix="_v2")

    print("\n================================================================================")
    print("MAJOR STEP 5 FINAL LOCKED TEST EVALUATION (v2) COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Final locked test evaluation")
    parser.add_argument("--v2", action="store_true", help="Evaluate 30-feature v2 scenario-a models")
    args = parser.parse_args()

    if args.v2:
        run_final_test_evaluation_v2()
    else:
        run_final_test_evaluation()
