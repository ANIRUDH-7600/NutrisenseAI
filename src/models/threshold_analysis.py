"""
Step 11: Decision Threshold Analysis, Probability Calibration, and Decision Curve Analysis.
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
Scenario A — Community Pre-Screening Setting

Performs a rigorous, leakage-safe post-estimation analysis on the Validation partition:
1. Evaluates a continuous decision-threshold grid (tau in [0.01, 0.99]) across models and targets.
2. Identifies analytical candidate thresholds (Max F1, Max Youden's J, Max Balanced Accuracy).
3. Investigates the trade-off between loss-reweighting and threshold-shifting.
4. Performs leakage-safe probability calibration (Platt scaling, Isotonic regression) fitted strictly on X_train.
5. Computes decision-analytic Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.

The Test set remains completely locked and unaccessed.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from lightgbm import LGBMClassifier

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.features.build_features import (
    transform_candidate_features,
    transform_candidate_features_v2,
    SCENARIO_A_V2_FEATURE_NAMES,
    validate_scenario_a_v2_feature_matrix
)
from src.models.baseline_model import build_preprocessing_pipeline

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745
INPUT_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
FEATURE_METADATA_V2_PATH = "data/interim/feature_metadata_v2.json"
MODELS_DIR = "models"
MODELS_V2_DIR = "models/v2"
FIGURES_DIR = "reports/figures"

OUTPUT_THRESHOLD_JSON = "data/interim/threshold_analysis_metrics.json"
OUTPUT_THRESHOLD_CSV = "data/interim/threshold_analysis_metrics.csv"
OUTPUT_THRESHOLD_V2_JSON = "data/interim/threshold_analysis_metrics_v2.json"
OUTPUT_THRESHOLD_V2_CSV = "data/interim/threshold_analysis_metrics_v2.csv"
OUTPUT_CALIBRATION_JSON = "data/interim/calibration_metrics.json"
OUTPUT_DCA_CSV = "data/interim/decision_curve_metrics.csv"

RANDOM_SEED = 42
TARGET_NAMES = ["stunting", "underweight", "wasting"]
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
DCA_THRESHOLD_RANGE = np.round(np.arange(0.05, 0.71, 0.02), 2)


def verify_raw_dataset_immutability():
    """Verify raw DHS data integrity."""
    if not os.path.exists(RAW_DTA_PATH):
        raise FileNotFoundError(f"Raw dataset {RAW_DTA_PATH} not found.")
    actual_bytes = os.path.getsize(RAW_DTA_PATH)
    if actual_bytes != EXPECTED_RAW_BYTES:
        raise ValueError(
            f"Raw dataset altered! Expected {EXPECTED_RAW_BYTES} bytes, found {actual_bytes} bytes."
        )
    print(f"[OK] Raw dataset immutability verified ({actual_bytes:,} bytes).")


def evaluate_threshold_point(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict:
    """Computes all diagnostic and screening metrics at a given threshold tau."""
    y_pred = (y_prob >= tau).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fp + fn + tn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    ba = (sens + spec) / 2.0
    youden_j = sens + spec - 1.0
    fpr = 1.0 - spec
    fnr = 1.0 - sens

    return {
        "threshold": round(float(tau), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "balanced_accuracy": round(float(ba), 4),
        "youden_j": round(float(youden_j), 4),
        "npv": round(float(npv), 4),
        "ppv": round(float(prec), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4)
    }


def compute_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, pt: float) -> dict:
    """Computes Decision Curve Analysis net benefit for a given decision threshold pt."""
    N = len(y_true)
    y_pred = (y_prob >= pt).astype(int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())

    weight = pt / (1.0 - pt)
    nb_model = (tp / N) - (fp / N) * weight

    # Treat-all strategy
    prev = float((y_true == 1).sum()) / N
    nb_all = prev - (1.0 - prev) * weight

    # Treat-none strategy
    nb_none = 0.0

    return {
        "threshold_probability": round(float(pt), 2),
        "net_benefit_model": round(float(nb_model), 5),
        "net_benefit_all": round(float(nb_all), 5),
        "net_benefit_none": round(float(nb_none), 5)
    }


def run_threshold_analysis_v1():
    print("=" * 70)
    print("NutriSense AI — Step 11: Threshold Calibration & DCA Pipeline")
    print("=" * 70)

    # 1. Immutability audit
    verify_raw_dataset_immutability()

    # 2. Load dataset and feature registry
    print(f"\nLoading cleaned data from: {INPUT_DATA_PATH}")
    df_raw = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
    assert "split" in df_raw.columns, "Missing 'split' column! Step 8 required."

    with open(FEATURE_METADATA_PATH, "r") as f:
        feature_meta = json.load(f)
    approved_features = list(feature_meta["feature_statistics"].keys())
    assert len(approved_features) == 34

    print("Extracting 34 Scenario A candidate features...")
    X_all = transform_candidate_features(df_raw)

    # 3. Model set to analyze
    # Evaluates LR (unweighted), LR (class-weighted), LightGBM (unweighted), LightGBM (class-weighted)
    model_configs = [
        ("Logistic Regression (Unweighted)", "baseline_logistic_{target}.joblib"),
        ("Logistic Regression (Class-Weighted)", "model_comparison_logistic_regression_{target}_class_weighted.joblib"),
        ("LightGBM (Unweighted)", "model_comparison_lightgbm_{target}_unweighted.joblib"),
        ("LightGBM (Class-Weighted)", "model_comparison_lightgbm_{target}_class_weighted.joblib")
    ]

    all_grid_records = []
    threshold_summary = {}
    calibration_summary = {}
    dca_records = []

    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Data structures for multi-target ROC and PR plotting
    roc_plot_data = {}
    pr_plot_data = {}
    calib_plot_data = {}
    dca_plot_data = {}

    for target_name in TARGET_NAMES:
        print(f"\n{'='*30} TARGET: {target_name.upper()} {'='*30}")
        valid_mask = df_raw[f"eligible_{target_name}"] == True
        train_mask = (df_raw["split"] == "train") & valid_mask
        val_mask = (df_raw["split"] == "val") & valid_mask

        X_val = X_all.loc[val_mask]
        y_val = df_raw.loc[val_mask, target_name].astype(int).values
        n_val = len(y_val)
        val_pos = int((y_val == 1).sum())
        val_prev = round((val_pos / n_val) * 100, 2)

        print(f"Validation Cohort: N = {n_val:,} (Pos: {val_pos:,}, Neg: {n_val - val_pos:,}, Prev: {val_prev:.2f}%)")

        threshold_summary[target_name] = {
            "validation_sample_size": n_val,
            "prevalence_pct": val_prev,
            "models": {}
        }

        # Store threshold evaluation curves for this target
        target_curve_dfs = {}

        for model_label, model_pattern in model_configs:
            model_file = os.path.join(MODELS_DIR, model_pattern.format(target=target_name))
            assert os.path.exists(model_file), f"Missing model artifact: {model_file}"

            pipeline = joblib.load(model_file)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            # Compute full threshold grid
            grid_points = [evaluate_threshold_point(y_val, y_prob, tau) for tau in THRESHOLD_GRID]
            df_grid = pd.DataFrame(grid_points)
            df_grid["target"] = target_name
            df_grid["model"] = model_label

            all_grid_records.extend(df_grid.to_dict("records"))
            target_curve_dfs[model_label] = df_grid

            # Identify analytical criteria
            idx_max_f1 = df_grid["f1_score"].idxmax()
            best_f1_point = df_grid.loc[idx_max_f1].to_dict()

            idx_max_j = df_grid["youden_j"].idxmax()
            best_j_point = df_grid.loc[idx_max_j].to_dict()

            idx_max_ba = df_grid["balanced_accuracy"].idxmax()
            best_ba_point = df_grid.loc[idx_max_ba].to_dict()

            # Point at default 0.50
            idx_05 = (df_grid["threshold"] - 0.50).abs().idxmin()
            point_05 = df_grid.loc[idx_05].to_dict()

            # Overall ranking metrics
            roc_auc = round(float(roc_auc_score(y_val, y_prob)), 4)
            pr_auc = round(float(average_precision_score(y_val, y_prob)), 4)
            brier_uncal = round(float(brier_score_loss(y_val, y_prob)), 4)

            print(f"  {model_label:<38} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier_uncal:.4f}")
            print(f"    - Default (tau=0.50): Sens={point_05['recall_sensitivity']:.4f}, Spec={point_05['specificity']:.4f}, Prec={point_05['precision']:.4f}, F1={point_05['f1_score']:.4f}")
            print(f"    - Max F1 (tau={best_f1_point['threshold']:.2f}): Sens={best_f1_point['recall_sensitivity']:.4f}, Spec={best_f1_point['specificity']:.4f}, Prec={best_f1_point['precision']:.4f}, F1={best_f1_point['f1_score']:.4f}")
            print(f"    - Max Youden J (tau={best_j_point['threshold']:.2f}): Sens={best_j_point['recall_sensitivity']:.4f}, Spec={best_j_point['specificity']:.4f}, Youden_J={best_j_point['youden_j']:.4f}")

            threshold_summary[target_name]["models"][model_label] = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score_uncalibrated": brier_uncal,
                "default_threshold_0_5": point_05,
                "max_f1_threshold": best_f1_point,
                "max_youden_j_threshold": best_j_point,
                "max_balanced_accuracy_threshold": best_ba_point
            }

            # Save ROC and PR curve coordinates for LightGBM and LR unweighted
            if model_label in ["Logistic Regression (Unweighted)", "LightGBM (Unweighted)"]:
                fpr_vals, tpr_vals, _ = roc_curve(y_val, y_prob)
                p_vals, r_vals, _ = precision_recall_curve(y_val, y_prob)
                roc_plot_data[(target_name, model_label)] = (fpr_vals, tpr_vals, roc_auc)
                pr_plot_data[(target_name, model_label)] = (r_vals, p_vals, pr_auc, val_prev / 100.0)

            # Decision Curve Analysis for LightGBM Unweighted
            if model_label == "LightGBM (Unweighted)":
                dca_pts = [compute_net_benefit(y_val, y_prob, pt) for pt in DCA_THRESHOLD_RANGE]
                df_dca = pd.DataFrame(dca_pts)
                df_dca["target"] = target_name
                df_dca["model"] = model_label
                dca_records.extend(df_dca.to_dict("records"))
                dca_plot_data[target_name] = df_dca

        # Generate target-specific threshold curve figure (Fig 12, 13, 14)
        fig_num = 12 if target_name == "stunting" else (13 if target_name == "underweight" else 14)
        fig_path = os.path.join(FIGURES_DIR, f"fig{fig_num}_threshold_curves_{target_name}.png")

        df_lgb_unw = target_curve_dfs["LightGBM (Unweighted)"]
        plt.figure(figsize=(10, 6))
        plt.plot(df_lgb_unw["threshold"], df_lgb_unw["recall_sensitivity"], label="Recall (Sensitivity)", color="#C44E52", linewidth=2.5)
        plt.plot(df_lgb_unw["threshold"], df_lgb_unw["specificity"], label="Specificity", color="#4C72B0", linewidth=2.5)
        plt.plot(df_lgb_unw["threshold"], df_lgb_unw["precision"], label="Precision (PPV)", color="#55A868", linewidth=2.5)
        plt.plot(df_lgb_unw["threshold"], df_lgb_unw["f1_score"], label="F1-Score", color="#8172B3", linewidth=2.5, linestyle="--")

        # Mark Max F1 and Youden J thresholds
        opt_f1_t = threshold_summary[target_name]["models"]["LightGBM (Unweighted)"]["max_f1_threshold"]["threshold"]
        opt_j_t = threshold_summary[target_name]["models"]["LightGBM (Unweighted)"]["max_youden_j_threshold"]["threshold"]

        plt.axvline(opt_f1_t, color="#8172B3", linestyle=":", label=f"Max F1 ({opt_f1_t:.2f})")
        plt.axvline(opt_j_t, color="#4C72B0", linestyle=":", label=f"Max Youden's J ({opt_j_t:.2f})")
        plt.axvline(0.50, color="gray", linestyle="-.", alpha=0.7, label="Default (0.50)")

        plt.title(f"Figure {fig_num}: Screening Metric Trade-offs Across Decision Thresholds ({target_name.capitalize()} — LightGBM)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Decision Threshold (tau)", fontsize=11, labelpad=8)
        plt.ylabel("Metric Score", fontsize=11)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.02)
        plt.legend(loc="center right", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"  [OK] Saved threshold curve figure: {fig_path}")

        # 4. Probability Calibration (LightGBM on X_train evaluated on X_val)
        print(f"  Evaluating probability calibration for {target_name}...")
        X_train = X_all.loc[train_mask]
        y_train = df_raw.loc[train_mask, target_name].astype(int).values

        preprocessor, _, _ = build_preprocessing_pipeline(approved_features)
        base_lgbm = LGBMClassifier(
            n_estimators=100, max_depth=6, num_leaves=31, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
        )

        # A. Uncalibrated
        pipe_uncal = Pipeline([("preprocessor", preprocessor), ("classifier", base_lgbm)])
        pipe_uncal.fit(X_train, y_train)
        prob_uncal = pipe_uncal.predict_proba(X_val)[:, 1]
        brier_uncal = round(float(brier_score_loss(y_val, prob_uncal)), 5)
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_val, prob_uncal, n_bins=10, strategy="uniform")

        # B. Platt Scaling (Sigmoid CalibratedClassifierCV fitted strictly on train via 3-fold CV)
        cal_platt = CalibratedClassifierCV(estimator=pipe_uncal, method="sigmoid", cv=3)
        cal_platt.fit(X_train, y_train)
        prob_platt = cal_platt.predict_proba(X_val)[:, 1]
        brier_platt = round(float(brier_score_loss(y_val, prob_platt)), 5)
        prob_true_platt, prob_pred_platt = calibration_curve(y_val, prob_platt, n_bins=10, strategy="uniform")

        # C. Isotonic Regression CalibratedClassifierCV fitted strictly on train
        cal_iso = CalibratedClassifierCV(estimator=pipe_uncal, method="isotonic", cv=3)
        cal_iso.fit(X_train, y_train)
        prob_iso = cal_iso.predict_proba(X_val)[:, 1]
        brier_iso = round(float(brier_score_loss(y_val, prob_iso)), 5)
        prob_true_iso, prob_pred_iso = calibration_curve(y_val, prob_iso, n_bins=10, strategy="uniform")

        print(f"    Calibration Brier Scores: Uncalibrated={brier_uncal:.5f} | Platt={brier_platt:.5f} | Isotonic={brier_iso:.5f}")

        calibration_summary[target_name] = {
            "validation_sample_size": n_val,
            "prevalence": val_prev / 100.0,
            "brier_scores": {
                "uncalibrated": brier_uncal,
                "platt_scaling": brier_platt,
                "isotonic_regression": brier_iso
            },
            "calibration_curve_points": {
                "uncalibrated": {"prob_pred": prob_pred_uncal.tolist(), "prob_true": prob_true_uncal.tolist()},
                "platt_scaling": {"prob_pred": prob_pred_platt.tolist(), "prob_true": prob_true_platt.tolist()},
                "isotonic_regression": {"prob_pred": prob_pred_iso.tolist(), "prob_true": prob_true_iso.tolist()}
            }
        }

        calib_plot_data[target_name] = {
            "uncal": (prob_pred_uncal, prob_true_uncal, brier_uncal),
            "platt": (prob_pred_platt, prob_true_platt, brier_platt),
            "iso": (prob_pred_iso, prob_true_iso, brier_iso)
        }

    # 5. Generate Multi-Target Comparison Figures (Fig 15, 16, 17, 18)

    # Figure 15: Precision-Recall Curves
    plt.figure(figsize=(11, 7))
    for (t_name, m_name), (r_vals, p_vals, pr_auc, prev) in pr_plot_data.items():
        ls = "-" if "LightGBM" in m_name else "--"
        label = f"{t_name.capitalize()} ({m_name.split()[0]}) — PR-AUC: {pr_auc:.3f} (Prev: {prev*100:.1f}%)"
        plt.plot(r_vals, p_vals, label=label, linestyle=ls, linewidth=2.0)
    plt.title("Figure 15: Precision-Recall Curves Across Nutritional Targets and Model Families", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall (Sensitivity)", fontsize=11, labelpad=8)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    fig15_path = os.path.join(FIGURES_DIR, "fig15_precision_recall_curves.png")
    plt.savefig(fig15_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig15_path}")

    # Figure 16: ROC Curves
    plt.figure(figsize=(10, 7))
    for (t_name, m_name), (fpr_vals, tpr_vals, roc_auc) in roc_plot_data.items():
        ls = "-" if "LightGBM" in m_name else "--"
        label = f"{t_name.capitalize()} ({m_name.split()[0]}) — ROC-AUC: {roc_auc:.4f}"
        plt.plot(fpr_vals, tpr_vals, label=label, linestyle=ls, linewidth=2.0)
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Chance Baseline (AUC: 0.50)")
    plt.title("Figure 16: Validation ROC Curves Across Nutritional Targets and Model Families", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig16_path = os.path.join(FIGURES_DIR, "fig16_roc_curves.png")
    plt.savefig(fig16_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig16_path}")

    # Figure 17: Probability Calibration Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, t_name in enumerate(TARGET_NAMES):
        ax = axes[idx]
        cd = calib_plot_data[t_name]
        ax.plot([0, 1], [0, 1], color="black", linestyle=":", label="Perfect Calibration")
        ax.plot(cd["uncal"][0], cd["uncal"][1], marker="o", label=f"Uncalibrated (Brier: {cd['uncal'][2]:.4f})", color="#C44E52")
        ax.plot(cd["platt"][0], cd["platt"][1], marker="s", label=f"Platt Scaling (Brier: {cd['platt'][2]:.4f})", color="#4C72B0")
        ax.plot(cd["iso"][0], cd["iso"][1], marker="^", label=f"Isotonic (Brier: {cd['iso'][2]:.4f})", color="#55A868")
        ax.set_title(f"{t_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Observed Event Frequency", fontsize=10)
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(0.0, 0.8)
        ax.legend(fontsize=8.5, loc="upper left")
    plt.suptitle("Figure 17: Reliability Diagrams and Probability Calibration Across Targets (LightGBM)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig17_path = os.path.join(FIGURES_DIR, "fig17_probability_calibration_curves.png")
    plt.savefig(fig17_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {fig17_path}")

    # Figure 18: Decision Curve Analysis (Net Benefit)
    plt.figure(figsize=(11, 6))
    colors = {"stunting": "#4C72B0", "underweight": "#55A868", "wasting": "#C44E52"}
    for t_name, df_dca in dca_plot_data.items():
        c = colors[t_name]
        plt.plot(df_dca["threshold_probability"], df_dca["net_benefit_model"], label=f"{t_name.capitalize()} (Screening Model)", color=c, linewidth=2.5)
        plt.plot(df_dca["threshold_probability"], df_dca["net_benefit_all"], label=f"{t_name.capitalize()} (Treat All)", color=c, linestyle="--", alpha=0.5)
    plt.axhline(0, color="black", linestyle=":", label="Treat None (Net Benefit = 0)")
    plt.title("Figure 18: Decision Curve Analysis (DCA) Net Benefit Across Threshold Probabilities", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Threshold Probability (pt)", fontsize=11, labelpad=8)
    plt.ylabel("Decision-Analytic Net Benefit", fontsize=11)
    plt.xlim(0.05, 0.65)
    plt.ylim(-0.05, 0.40)
    plt.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    fig18_path = os.path.join(FIGURES_DIR, "fig18_decision_curve_analysis.png")
    plt.savefig(fig18_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig18_path}")

    # 6. Save Data Artifacts
    df_all_grid = pd.DataFrame(all_grid_records)
    df_all_grid.to_csv(OUTPUT_THRESHOLD_CSV, index=False)
    print(f"\n[OK] Threshold analysis CSV saved: {OUTPUT_THRESHOLD_CSV}")

    with open(OUTPUT_THRESHOLD_JSON, "w") as f:
        json.dump({
            "step": "Step 11 — Threshold Calibration & Operational DCA",
            "scenario": "Scenario A — Community Pre-Screening",
            "test_set_status": {
                "locked": True,
                "accessed": False,
                "statement": "The test set was not used for threshold grid evaluation, candidate threshold selection, calibration fitting, or DCA calculations."
            },
            "threshold_summary": threshold_summary
        }, f, indent=2)
    print(f"[OK] Threshold analysis JSON saved: {OUTPUT_THRESHOLD_JSON}")

    with open(OUTPUT_CALIBRATION_JSON, "w") as f:
        json.dump({
            "step": "Step 11 — Probability Calibration",
            "methodology": "CalibratedClassifierCV (Platt scaling and Isotonic regression) fitted strictly on X_train via 3-fold CV",
            "calibration_summary": calibration_summary
        }, f, indent=2)
    print(f"[OK] Calibration metrics JSON saved: {OUTPUT_CALIBRATION_JSON}")

    df_dca_all = pd.DataFrame(dca_records)
    df_dca_all.to_csv(OUTPUT_DCA_CSV, index=False)
    print(f"[OK] Decision Curve Analysis CSV saved: {OUTPUT_DCA_CSV}")

    print("\n" + "=" * 70)
    print("Step 11: Threshold Analysis, Calibration, and DCA completed successfully!")
    print("=" * 70)


def run_threshold_analysis_v2():
    print("=" * 70)
    print("NutriSense AI — Threshold Analysis (30-Feature Scenario-A v2)")
    print("=" * 70)

    # 1. Immutability audit
    verify_raw_dataset_immutability()

    # 2. Load dataset and v2 feature registry
    print(f"\nLoading cleaned data from: {INPUT_DATA_PATH}")
    df_raw = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
    assert "split" in df_raw.columns

    with open(FEATURE_METADATA_V2_PATH, "r") as f:
        feature_meta_v2 = json.load(f)
    approved_features = feature_meta_v2["features"]
    assert len(approved_features) == 30

    print("Extracting 30 Scenario A candidate features...")
    X_all = transform_candidate_features_v2(df_raw)
    validate_scenario_a_v2_feature_matrix(X_all)

    # 3. Model set to analyze on 30 features
    model_configs = [
        ("LightGBM (Unweighted)", "model_comparison_lightgbm_{target}_unweighted_v2.joblib"),
        ("LightGBM (Class-Weighted)", "model_comparison_lightgbm_{target}_class_weighted_v2.joblib"),
        ("XGBoost (Unweighted)", "model_comparison_xgboost_{target}_unweighted_v2.joblib"),
        ("XGBoost (Class-Weighted)", "model_comparison_xgboost_{target}_class_weighted_v2.joblib"),
        ("Logistic Regression (Unweighted)", "baseline_logistic_{target}_v2.joblib"),
        ("Logistic Regression (Class-Weighted)", "model_comparison_logistic_regression_{target}_class_weighted_v2.joblib")
    ]

    all_grid_records = []
    threshold_summary = {}

    for target_name in TARGET_NAMES:
        print(f"\n{'='*30} TARGET: {target_name.upper()} (30 Features v2) {'='*30}")
        valid_mask = df_raw[f"eligible_{target_name}"] == True
        val_mask = (df_raw["split"] == "val") & valid_mask

        X_val = X_all.loc[val_mask]
        y_val = df_raw.loc[val_mask, target_name].astype(int).values
        n_val = len(y_val)
        val_pos = int((y_val == 1).sum())
        val_prev = round((val_pos / n_val) * 100, 2)

        print(f"Validation Cohort: N = {n_val:,} (Pos: {val_pos:,}, Neg: {n_val - val_pos:,}, Prev: {val_prev:.2f}%)")

        threshold_summary[target_name] = {
            "validation_sample_size": n_val,
            "prevalence_pct": val_prev,
            "models": {}
        }

        for model_label, model_pattern in model_configs:
            model_file = os.path.join(MODELS_V2_DIR, model_pattern.format(target=target_name))
            assert os.path.exists(model_file), f"Missing model artifact: {model_file}"

            pipeline = joblib.load(model_file)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            # Compute full threshold grid
            grid_points = [evaluate_threshold_point(y_val, y_prob, tau) for tau in THRESHOLD_GRID]
            df_grid = pd.DataFrame(grid_points)
            df_grid["target"] = target_name
            df_grid["model"] = model_label

            all_grid_records.extend(df_grid.to_dict("records"))

            # Criteria points
            idx_max_f1 = df_grid["f1_score"].idxmax()
            best_f1_point = df_grid.loc[idx_max_f1].to_dict()

            idx_max_j = df_grid["youden_j"].idxmax()
            best_j_point = df_grid.loc[idx_max_j].to_dict()

            idx_max_ba = df_grid["balanced_accuracy"].idxmax()
            best_ba_point = df_grid.loc[idx_max_ba].to_dict()

            idx_05 = (df_grid["threshold"] - 0.50).abs().idxmin()
            point_05 = df_grid.loc[idx_05].to_dict()

            # Screening targeted threshold:
            # For Stunting & Underweight: Sensitivity >= 60%, Specificity >= 50%
            # For Wasting: Sensitivity >= 65%, Specificity >= 45%
            if target_name in ["stunting", "underweight"]:
                valid_triage = df_grid[(df_grid["recall_sensitivity"] >= 0.60) & (df_grid["specificity"] >= 0.50)]
            else:
                valid_triage = df_grid[(df_grid["recall_sensitivity"] >= 0.65) & (df_grid["specificity"] >= 0.45)]

            if len(valid_triage) > 0:
                idx_triage = valid_triage["balanced_accuracy"].idxmax()
                screening_point = valid_triage.loc[idx_triage].to_dict()
            else:
                screening_point = best_j_point

            roc_auc = round(float(roc_auc_score(y_val, y_prob)), 4)
            pr_auc = round(float(average_precision_score(y_val, y_prob)), 4)
            brier_uncal = round(float(brier_score_loss(y_val, y_prob)), 4)

            print(f"  {model_label:<38} | ROC: {roc_auc:.4f} | PR: {pr_auc:.4f} | Brier: {brier_uncal:.4f}")
            print(f"    - Default (tau=0.50): Sens={point_05['recall_sensitivity']:.4f}, Spec={point_05['specificity']:.4f}, Prec={point_05['precision']:.4f}, F1={point_05['f1_score']:.4f}")
            print(f"    - Max F1 (tau={best_f1_point['threshold']:.2f}): Sens={best_f1_point['recall_sensitivity']:.4f}, Spec={best_f1_point['specificity']:.4f}, Prec={best_f1_point['precision']:.4f}, F1={best_f1_point['f1_score']:.4f}")
            print(f"    - Max Youden J (tau={best_j_point['threshold']:.2f}): Sens={best_j_point['recall_sensitivity']:.4f}, Spec={best_j_point['specificity']:.4f}, Youden_J={best_j_point['youden_j']:.4f}")
            print(f"    - Operating Screening Point (tau={screening_point['threshold']:.2f}): Sens={screening_point['recall_sensitivity']:.4f}, Spec={screening_point['specificity']:.4f}, F1={screening_point['f1_score']:.4f}")

            threshold_summary[target_name]["models"][model_label] = {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier_score_uncalibrated": brier_uncal,
                "default_threshold_0_5": point_05,
                "max_f1_threshold": best_f1_point,
                "max_youden_j_threshold": best_j_point,
                "max_balanced_accuracy_threshold": best_ba_point,
                "validation_operating_threshold": screening_point
            }

    # Save Data Artifacts
    df_all_grid = pd.DataFrame(all_grid_records)
    os.makedirs(os.path.dirname(OUTPUT_THRESHOLD_V2_CSV), exist_ok=True)
    df_all_grid.to_csv(OUTPUT_THRESHOLD_V2_CSV, index=False)
    print(f"\n[OK] Threshold analysis CSV saved: {OUTPUT_THRESHOLD_V2_CSV}")

    with open(OUTPUT_THRESHOLD_V2_JSON, "w") as f:
        json.dump({
            "step": "Threshold Analysis (30-Feature Scenario-A v2)",
            "scenario": "Scenario A — Community Pre-Screening",
            "feature_count": 30,
            "terminology_note": "Thresholds are validation-derived operating thresholds (not probability calibrated).",
            "test_set_status": {
                "locked": True,
                "accessed": False,
                "statement": "The test set was not used for threshold grid evaluation, candidate threshold selection, or model selection."
            },
            "threshold_summary": threshold_summary
        }, f, indent=2)
    print(f"[OK] Threshold analysis JSON saved: {OUTPUT_THRESHOLD_V2_JSON}")

    print("\n" + "=" * 70)
    print("Threshold Analysis (30-feature v2) completed successfully!")
    print("=" * 70)
    return threshold_summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Threshold Analysis for Scenario A.")
    parser.add_argument("--v2", action="store_true", help="Run 30-feature v2 threshold analysis.")
    parser.add_argument("--all", action="store_true", help="Run both v1 and v2 threshold analysis.")
    args = parser.parse_args()

    if args.v2:
        run_threshold_analysis_v2()
        return

    run_threshold_analysis_v1()
    if args.all:
        run_threshold_analysis_v2()


if __name__ == "__main__":
    main()
