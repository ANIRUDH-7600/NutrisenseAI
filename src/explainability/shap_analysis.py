#!/usr/bin/env python3
"""
Step 14: Model Explainability & Interpretability using TreeSHAP
Project: NutriSense AI — Multimodal Childhood Malnutrition Risk Intelligence
Dataset: NFHS-5 India 2019–21 Children's Recode (KR)
Setting: Scenario A — Community Pre-Screening (34 non-invasive features)

Applies TreeSHAP (shap.TreeExplainer) to explain the pre-selected LightGBM Unweighted
models across Stunting, Underweight, and Wasting.

CRITICAL GOVERNANCE & PRIVACY RULES:
1. Purely interpretability: zero model retraining, zero hyperparameter tuning,
   zero feature selection, and zero threshold modification.
2. Explanations derived exclusively from a reproducible sample (N = 5,000, seed = 42)
   of the VALIDATION partition to avoid post-test analytical flexibility.
3. Scenario A boundary: exactly the 34 approved candidate features;
   prohibits all direct anthropometrics (hw2-hw12, hw70-hw73, hw13, targets).
4. Strict data privacy: NO individual DHS child records, household IDs, or individual
   SHAP vectors are saved to public artifacts. All saved artifacts contain strictly
   aggregate model-level summaries.
5. Raw DHS dataset immutability verified (441,380,745 bytes).
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.features.build_features import transform_candidate_features, transform_candidate_features_v2

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745

DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
FEATURE_METADATA_PATH_V2 = "data/interim/feature_metadata_v2.json"
MODELS_DIR = "models"
MODELS_DIR_V2 = "models/v2"
FIGURES_DIR = "reports/figures"

OUTPUT_SHAP_CSV = "data/interim/shap_feature_importance.csv"
OUTPUT_SHAP_JSON = "data/interim/shap_feature_importance.json"
OUTPUT_CROSS_TARGET_CSV = "data/interim/shap_cross_target_comparison.csv"
OUTPUT_TOP_FEATURES_JSON = "data/interim/shap_top_features.json"

OUTPUT_SHAP_CSV_V2 = "data/interim/shap_feature_importance_v2.csv"
OUTPUT_SHAP_JSON_V2 = "data/interim/shap_feature_importance_v2.json"
OUTPUT_CROSS_TARGET_CSV_V2 = "data/interim/shap_cross_target_comparison_v2.csv"
OUTPUT_TOP_FEATURES_JSON_V2 = "data/interim/shap_top_features_v2.json"

RANDOM_SEED = 42
SHAP_SAMPLE_SIZE = 5000
INTERACTION_SAMPLE_SIZE = 250

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

TARGET_CONFIGS_V2 = {
    "stunting": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_stunting_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.36,
        "eligibility_col": "eligible_stunting"
    },
    "underweight": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_underweight_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.30,
        "eligibility_col": "eligible_underweight"
    },
    "wasting": {
        "model_file": os.path.join(MODELS_DIR_V2, "model_comparison_lightgbm_wasting_unweighted_v2.joblib"),
        "model_id": "LightGBM Unweighted",
        "threshold": 0.17,
        "eligibility_col": "eligible_wasting"
    }
}


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


def aggregate_transformed_shap_to_original(sv_transformed, transformed_cols, original_features):
    """
    Aggregates one-hot encoded categorical SHAP columns back to their parent candidate features.
    
    For numerical features, the mapping is 1-to-1:
        phi_{i, num_feat} = sv_{i, col}
    For one-hot categorical features, the observation-level contribution is the sum
    across mutually exclusive indicator levels:
        phi_{i, cat_feat} = sum_{k in levels} sv_{i, col_k}
    
    By Shapley linearity and additivity, the sum of phi_{i, F} across all 34 candidate
    features exactly equals the model log-odds deviation:
        sum_{F=1}^{34} phi_{i, F} = f(x_i) - E[f(X)].
    """
    n_samples = sv_transformed.shape[0]
    n_orig = len(original_features)
    sv_original = np.zeros((n_samples, n_orig), dtype=float)

    for orig_idx, orig_col in enumerate(original_features):
        matching_indices = [
            t_idx for t_idx, t_col in enumerate(transformed_cols)
            if t_col == f"num__{orig_col}" or t_col.startswith(f"cat__{orig_col}_")
        ]
        if not matching_indices:
            raise ValueError(f"Feature '{orig_col}' could not be mapped to any transformed column!")
        sv_original[:, orig_idx] = sv_transformed[:, matching_indices].sum(axis=1)

    return sv_original


def determine_feature_direction(feature_values, shap_values):
    """
    Computes empirical Spearman rank correlation to describe the directional association
    between feature values and model-assigned SHAP log-odds contributions.
    """
    # Filter out NaNs if any
    valid_mask = ~np.isnan(feature_values) & ~np.isnan(shap_values)
    x_valid = feature_values[valid_mask]
    s_valid = shap_values[valid_mask]

    if len(np.unique(x_valid)) <= 1 or len(np.unique(s_valid)) <= 1:
        return 0.0, "neutral"

    corr = np.corrcoef(pd.Series(x_valid).rank(), pd.Series(s_valid).rank())[0, 1]
    if np.isnan(corr):
        return 0.0, "neutral"

    if corr >= 0.15:
        dir_label = "positive"
    elif corr <= -0.15:
        dir_label = "negative"
    else:
        dir_label = "nonlinear_or_complex"

    return float(corr), dir_label


def generate_shap_visualizations(
    target,
    sv_34,
    X_sample,
    original_features,
    mean_abs_shap,
    ranked_indices,
    suffix=""
):
    """Generates summary beeswarm, global bar, and dependence plots."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Encode categorical features numerically for summary colorbar rendering
    X_numeric = X_sample.copy()
    for col in X_numeric.columns:
        if not pd.api.types.is_numeric_dtype(X_numeric[col]):
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes

    # 1. SHAP Beeswarm Summary Plot (Top 15 features)
    fig_summary_path = os.path.join(FIGURES_DIR, f"fig{23 + list(TARGET_CONFIGS.keys()).index(target)}_shap_summary_{target}{suffix}.png")
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        sv_34,
        features=X_numeric,
        feature_names=original_features,
        max_display=15,
        show=False,
        plot_size=None
    )
    title_suffix = " (30-Feature v2)" if suffix == "_v2" else ""
    plt.title(f"SHAP Beeswarm Summary Plot — {target.capitalize()} (LightGBM Unweighted{title_suffix}, N = {SHAP_SAMPLE_SIZE:,})",
              fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("SHAP Value (Impact on Model Log-Odds Output)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Beeswarm Summary: {fig_summary_path}")

    # 2. SHAP Global Feature Importance Bar Plot (Top 15 features)
    fig_bar_path = os.path.join(FIGURES_DIR, f"fig{26 + list(TARGET_CONFIGS.keys()).index(target)}_shap_bar_{target}{suffix}.png")
    top15_indices = ranked_indices[:15]
    top15_names = [original_features[i] for i in top15_indices][::-1]
    top15_scores = [mean_abs_shap[i] for i in top15_indices][::-1]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(range(len(top15_names)), top15_scores, color="#2b5c8f", edgecolor="#1a365d", height=0.65)
    plt.yticks(range(len(top15_names)), top15_names, fontsize=10)
    plt.xlabel("Mean Absolute SHAP Value (mean(|SHAP|))", fontsize=11, fontweight="bold")
    plt.title(f"Top 15 Global Features by Contribution Magnitude — {target.capitalize()}{title_suffix}",
              fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.003, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                 va="center", ha="left", fontsize=9, color="#2d3748")

    plt.xlim(0, max(top15_scores) * 1.15)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Global Bar Plot: {fig_bar_path}")

    # 3. Methodologically Appropriate Dependence Plot for Top Continuous Feature
    continuous_candidates = [
        "child_age_months", "birth_weight_kg", "mother_bmi",
        "mother_age_years", "household_size", "wealth_quintile"
    ]
    top_cont_feat = None
    for idx in ranked_indices:
        feat_name = original_features[idx]
        if feat_name in continuous_candidates:
            top_cont_feat = feat_name
            break

    if top_cont_feat:
        fig_dep_path = os.path.join(FIGURES_DIR, f"fig{30 + list(TARGET_CONFIGS.keys()).index(target)}_shap_dependence_{target}{suffix}.png")
        cont_idx = original_features.index(top_cont_feat)
        feat_x = X_sample[top_cont_feat].values
        feat_shap = sv_34[:, cont_idx]

        plt.figure(figsize=(9, 6))
        scatter = plt.scatter(
            feat_x, feat_shap,
            c=X_numeric["wealth_quintile"],
            cmap="viridis",
            alpha=0.45,
            edgecolors="none",
            s=22
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Household Wealth Quintile (1=Poorest to 5=Richest)", fontsize=10)

        # Smooth lowess trend for visualization
        valid_dep = ~np.isnan(feat_x) & ~np.isnan(feat_shap)
        if np.sum(valid_dep) > 50:
            df_trend = pd.DataFrame({"x": feat_x[valid_dep], "y": feat_shap[valid_dep]}).sort_values("x")
            df_trend["y_smooth"] = df_trend["y"].rolling(window=200, min_periods=20, center=True).mean()
            plt.plot(df_trend["x"], df_trend["y_smooth"], color="#c53030", linewidth=2.5, label="Empirical Moving Average")

        plt.axhline(0, color="gray", linestyle="--", alpha=0.7, linewidth=1)
        plt.xlabel(f"{top_cont_feat} (Observed Feature Value)", fontsize=11, fontweight="bold")
        plt.ylabel("SHAP Value (Contribution to Log-Odds)", fontsize=11, fontweight="bold")
        plt.title(f"SHAP Dependence: {top_cont_feat} vs. {target.capitalize()} Risk{title_suffix}",
                  fontsize=13, fontweight="bold", pad=12)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(fig_dep_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [+] Saved Dependence Plot: {fig_dep_path}")


def plot_cross_target_comparison(df_cross, suffix=""):
    """Plots cross-target comparison of feature importance across Stunting, Underweight, and Wasting."""
    fig_cross_path = os.path.join(FIGURES_DIR, f"fig29_shap_cross_target_comparison{suffix}.png")
    
    # Sort features by average rank across all three targets
    df_sorted = df_cross.sort_values("mean_rank_across_targets", ascending=True).head(15)

    features = df_sorted["feature"].tolist()[::-1]
    stunting_scores = df_sorted["mean_abs_shap_stunting"].tolist()[::-1]
    underweight_scores = df_sorted["mean_abs_shap_underweight"].tolist()[::-1]
    wasting_scores = df_sorted["mean_abs_shap_wasting"].tolist()[::-1]

    y = np.arange(len(features))
    height = 0.26

    plt.figure(figsize=(12, 9))
    plt.barh(y + height, stunting_scores, height=height, label="Stunting", color="#3182ce", edgecolor="#2b6cb0")
    plt.barh(y, underweight_scores, height=height, label="Underweight", color="#38a169", edgecolor="#2f855a")
    plt.barh(y - height, wasting_scores, height=height, label="Wasting", color="#dd6b20", edgecolor="#c05621")

    plt.yticks(y, features, fontsize=10)
    plt.xlabel("Mean Absolute SHAP Value (mean(|SHAP|))", fontsize=11, fontweight="bold")
    title_suffix = " (30-Feature v2)" if suffix == "_v2" else ""
    plt.title(f"Cross-Target SHAP Comparison — Top 15 Consensus Features{title_suffix}", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_cross_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Cross-Target Comparison Plot: {fig_cross_path}")


def run_exploratory_interactions(explainer, X_trans_sample, transformed_cols, original_features, top5_features):
    """
    Computes exploratory pairwise interaction values on a small reproducible validation sample
    strictly for the top 5 features.
    """
    print("    Computing exploratory pairwise interactions for top features...")
    try:
        interaction_values = explainer.shap_interaction_values(X_trans_sample)
        # interaction_values shape: (N, 89, 89)
        top_interactions = []
        for i_feat in top5_features:
            matching_i = [
                idx for idx, col in enumerate(transformed_cols)
                if col == f"num__{i_feat}" or col.startswith(f"cat__{i_feat}_")
            ]
            for j_feat in top5_features:
                if i_feat >= j_feat:
                    continue
                matching_j = [
                    idx for idx, col in enumerate(transformed_cols)
                    if col == f"num__{j_feat}" or col.startswith(f"cat__{j_feat}_")
                ]
                # Sum interaction magnitudes across corresponding sub-blocks
                pair_inter = np.abs(interaction_values[:, matching_i, :][:, :, matching_j]).sum(axis=(1, 2))
                mean_inter = float(np.mean(pair_inter))
                top_interactions.append({
                    "feature_1": i_feat,
                    "feature_2": j_feat,
                    "mean_abs_interaction": round(mean_inter, 5)
                })

        top_interactions.sort(key=lambda x: x["mean_abs_interaction"], reverse=True)
        return top_interactions[:5]
    except Exception as e:
        print(f"    [!] Warning: Interaction computation bypassed due to: {e}")
        return []


def run_shap_analysis():
    """Main execution orchestrator for Step 14 Explainability / SHAP."""
    start_time = time.time()
    print("=" * 80)
    print("STEP 14: MODEL EXPLAINABILITY & INTERPRETABILITY USING TREESHAP")
    print("=" * 80)

    # 1. Verify Raw Dataset Immutability
    verify_raw_dataset_immutability()

    # 2. Load Processed Dataset
    print(f"\n[*] Loading processed dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"    Loaded {len(df_raw):,} records. Partitions: {df_raw['split'].value_counts().to_dict()}")

    # 3. Transform candidate features (34 approved predictors)
    print("[*] Constructing Scenario A candidate feature matrix (34 non-invasive features)...")
    X_all = transform_candidate_features(df_raw)
    original_features = list(X_all.columns)
    assert len(original_features) == 34, f"Expected exactly 34 features, found {len(original_features)}!"
    print(f"[OK] Candidate feature matrix constructed: {len(original_features)} features.")

    shap_results_by_target = {}
    feature_importance_rows = []
    top_features_summary = {}

    for target, config in TARGET_CONFIGS.items():
        print("\n" + "-" * 75)
        print(f"EXPLAINING CHAMPION MODEL — TARGET: {target.upper()}")
        print("-" * 75)

        model_file = config["model_file"]
        elig_col = config["eligibility_col"]

        # Load fitted pipeline
        pipeline = joblib.load(model_file)
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed_cols = preprocessor.get_feature_names_out()

        # Filter strictly to the validation partition and target eligibility
        val_mask = (df_raw["split"] == "val") & (df_raw[elig_col] == True)
        df_val_target = df_raw.loc[val_mask]
        print(f"  Validation eligible records: {len(df_val_target):,}")

        # Reproducible random sample of exactly 5,000 validation records
        df_val_sample = df_val_target.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
        X_sample = transform_candidate_features(df_val_sample)
        print(f"  Sampled {len(X_sample):,} validation records (seed = {RANDOM_SEED}) for TreeSHAP.")

        # Transform features through preprocessor
        X_trans = preprocessor.transform(X_sample)

        # Initialize TreeExplainer
        explainer = shap.TreeExplainer(classifier)
        print(f"  Computing TreeSHAP values for {X_trans.shape[0]:,} samples x {X_trans.shape[1]} transformed columns...")
        t_start_shap = time.time()
        sv_transformed = explainer.shap_values(X_trans)
        print(f"  [OK] TreeSHAP completed in {time.time() - t_start_shap:.2f} seconds.")

        # Aggregate back to exact 34 candidate features
        sv_34 = aggregate_transformed_shap_to_original(sv_transformed, transformed_cols, original_features)
        assert sv_34.shape == (SHAP_SAMPLE_SIZE, 34)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(sv_34), axis=0)
        ranked_indices = np.argsort(-mean_abs_shap)

        target_features_data = []
        for rank, idx in enumerate(ranked_indices, start=1):
            feat_name = original_features[idx]
            feat_score = float(mean_abs_shap[idx])
            
            # Directional effect
            feat_vals = pd.Categorical(X_sample[feat_name]).codes if not pd.api.types.is_numeric_dtype(X_sample[feat_name]) else X_sample[feat_name].values.astype(float)
            corr, dir_label = determine_feature_direction(feat_vals, sv_34[:, idx])

            feat_record = {
                "rank": rank,
                "feature": feat_name,
                "mean_abs_shap": round(feat_score, 5),
                "spearman_corr_with_shap": round(corr, 4),
                "direction": dir_label
            }
            target_features_data.append(feat_record)

            feature_importance_rows.append({
                "target": target,
                "rank": rank,
                "feature": feat_name,
                "mean_abs_shap": round(feat_score, 5),
                "spearman_corr_with_shap": round(corr, 4),
                "direction": dir_label
            })

        shap_results_by_target[target] = {
            "model_id": config["model_id"],
            "threshold": config["threshold"],
            "sample_size": SHAP_SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "cohort": "validation",
            "base_value_log_odds": float(explainer.expected_value),
            "features": target_features_data
        }

        top5 = [rec["feature"] for rec in target_features_data[:5]]
        top10 = [rec["feature"] for rec in target_features_data[:10]]
        print(f"  Top 5 Features: {', '.join(top5)}")

        # Exploratory interaction analysis on top 5 features using 250 samples
        df_inter_sample = df_val_target.sample(n=INTERACTION_SAMPLE_SIZE, random_state=RANDOM_SEED)
        X_inter_sample = transform_candidate_features(df_inter_sample)
        X_inter_trans = preprocessor.transform(X_inter_sample)
        top_interactions = run_exploratory_interactions(explainer, X_inter_trans, transformed_cols, original_features, top5)

        top_features_summary[target] = {
            "top_5": top5,
            "top_10": top10,
            "top_5_details": target_features_data[:5],
            "top_interactions": top_interactions
        }

        # Generate target figures
        generate_shap_visualizations(
            target=target,
            sv_34=sv_34,
            X_sample=X_sample,
            original_features=original_features,
            mean_abs_shap=mean_abs_shap,
            ranked_indices=ranked_indices
        )

    # 4. Cross-Target Synthesis & Comparison Table
    print("\n[*] Constructing Cross-Target Synthesis...")
    cross_target_rows = []
    for feat in original_features:
        stunt_rec = next(r for r in shap_results_by_target["stunting"]["features"] if r["feature"] == feat)
        under_rec = next(r for r in shap_results_by_target["underweight"]["features"] if r["feature"] == feat)
        waste_rec = next(r for r in shap_results_by_target["wasting"]["features"] if r["feature"] == feat)

        mean_rank = (stunt_rec["rank"] + under_rec["rank"] + waste_rec["rank"]) / 3.0
        cross_target_rows.append({
            "feature": feat,
            "rank_stunting": stunt_rec["rank"],
            "rank_underweight": under_rec["rank"],
            "rank_wasting": waste_rec["rank"],
            "mean_rank_across_targets": round(mean_rank, 2),
            "mean_abs_shap_stunting": stunt_rec["mean_abs_shap"],
            "mean_abs_shap_underweight": under_rec["mean_abs_shap"],
            "mean_abs_shap_wasting": waste_rec["mean_abs_shap"],
            "direction_stunting": stunt_rec["direction"],
            "direction_underweight": under_rec["direction"],
            "direction_wasting": waste_rec["direction"]
        })

    df_cross = pd.DataFrame(cross_target_rows).sort_values("mean_rank_across_targets", ascending=True)
    df_cross.to_csv(OUTPUT_CROSS_TARGET_CSV, index=False)
    print(f"[+] Saved Cross-Target CSV: {OUTPUT_CROSS_TARGET_CSV}")

    # Plot Cross-Target Comparison Figure
    plot_cross_target_comparison(df_cross)

    # 5. Export Feature Importance CSV and JSON
    df_importance = pd.DataFrame(feature_importance_rows)
    df_importance.to_csv(OUTPUT_SHAP_CSV, index=False)
    print(f"[+] Saved SHAP Feature Importance CSV: {OUTPUT_SHAP_CSV}")

    output_json_data = {
        "step": "Step 14 — Model Explainability & Interpretability using TreeSHAP",
        "scenario": "Scenario A — Community Pre-Screening",
        "dataset": "NFHS-5 India 2019-21 Children's Recode (KR)",
        "explanation_method": "shap.TreeExplainer (TreeSHAP)",
        "explanation_cohort": "Validation partition (strictly held-out from training)",
        "sample_size_per_target": SHAP_SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "total_candidate_features": len(original_features),
        "results_by_target": shap_results_by_target,
        "cross_target_top_consensus": df_cross.head(10)[["feature", "mean_rank_across_targets"]].to_dict(orient="records")
    }

    with open(OUTPUT_SHAP_JSON, "w") as f:
        json.dump(output_json_data, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved SHAP Feature Importance JSON: {OUTPUT_SHAP_JSON}")

    with open(OUTPUT_TOP_FEATURES_JSON, "w") as f:
        json.dump(top_features_summary, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved SHAP Top Features JSON: {OUTPUT_TOP_FEATURES_JSON}")

    elapsed = time.time() - start_time
    print(f"\n[OK] Step 14 SHAP analysis executed successfully in {elapsed:.1f} seconds.")


def run_shap_analysis_v2():
    """Main execution orchestrator for Major Step 6: SHAP Explainability (30 Features v2)."""
    start_time = time.time()
    print("=" * 80)
    print("MAJOR STEP 6: MODEL EXPLAINABILITY & INTERPRETABILITY USING TREESHAP (30 PREDICTORS v2)")
    print("=" * 80)

    # 1. Verify Raw Dataset Immutability
    verify_raw_dataset_immutability()

    # 2. Load Processed Dataset
    print(f"\n[*] Loading processed dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"    Loaded {len(df_raw):,} records. Partitions: {df_raw['split'].value_counts().to_dict()}")

    # 3. Transform candidate features (30 approved predictors for v2)
    print("[*] Constructing Scenario A v2 candidate feature matrix (30 non-invasive features)...")
    X_all = transform_candidate_features_v2(df_raw)
    original_features = list(X_all.columns)
    assert len(original_features) == 30, f"Expected exactly 30 features, found {len(original_features)}!"

    # Ensure removed sensitive attributes and prohibited variables are absent
    removed_attributes = {"household_head_female", "caste_category", "religion_category", "mother_education_level"}
    found_removed = set(original_features).intersection(removed_attributes)
    assert len(found_removed) == 0, f"Fatal: Removed sensitive attributes found in v2 feature set: {found_removed}"

    prohibited_vars = {"child_height_cm", "child_weight_kg", "hw70", "hw71", "hw72", "hw73", "haz", "waz", "whz"}
    found_prohibited = set(original_features).intersection(prohibited_vars)
    assert len(found_prohibited) == 0, f"Fatal: Prohibited anthropometrics found in v2 feature set: {found_prohibited}"

    print(f"[OK] Candidate feature matrix constructed: {len(original_features)} features (30 non-invasive predictors).")

    shap_results_by_target = {}
    feature_importance_rows = []
    top_features_summary = {}

    for target, config in TARGET_CONFIGS_V2.items():
        print("\n" + "-" * 75)
        print(f"EXPLAINING CHAMPION MODEL (v2) — TARGET: {target.upper()}")
        print("-" * 75)

        model_file = config["model_file"]
        elig_col = config["eligibility_col"]

        # Load fitted pipeline
        pipeline = joblib.load(model_file)
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        transformed_cols = preprocessor.get_feature_names_out()

        # Filter strictly to the validation partition and target eligibility
        val_mask = (df_raw["split"] == "val") & (df_raw[elig_col] == True)
        df_val_target = df_raw.loc[val_mask]
        print(f"  Validation eligible records: {len(df_val_target):,}")

        # Reproducible random sample of exactly 5,000 validation records
        df_val_sample = df_val_target.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
        X_sample = transform_candidate_features_v2(df_val_sample)
        print(f"  Sampled {len(X_sample):,} validation records (seed = {RANDOM_SEED}) for TreeSHAP.")

        # Transform features through preprocessor
        X_trans = preprocessor.transform(X_sample)

        # Initialize TreeExplainer
        explainer = shap.TreeExplainer(classifier)
        print(f"  Computing TreeSHAP values for {X_trans.shape[0]:,} samples x {X_trans.shape[1]} transformed columns...")
        t_start_shap = time.time()
        sv_transformed = explainer.shap_values(X_trans)
        if isinstance(sv_transformed, list):
            sv_transformed = sv_transformed[1]
        print(f"  [OK] TreeSHAP completed in {time.time() - t_start_shap:.2f} seconds.")

        # Aggregate back to exact 30 candidate features
        sv_30 = aggregate_transformed_shap_to_original(sv_transformed, transformed_cols, original_features)
        assert sv_30.shape == (SHAP_SAMPLE_SIZE, 30)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(sv_30), axis=0)
        ranked_indices = np.argsort(-mean_abs_shap)

        target_features_data = []
        for rank, idx in enumerate(ranked_indices, start=1):
            feat_name = original_features[idx]
            feat_score = float(mean_abs_shap[idx])
            
            # Directional effect
            feat_vals = pd.Categorical(X_sample[feat_name]).codes if not pd.api.types.is_numeric_dtype(X_sample[feat_name]) else X_sample[feat_name].values.astype(float)
            corr, dir_label = determine_feature_direction(feat_vals, sv_30[:, idx])

            feat_record = {
                "rank": rank,
                "feature": feat_name,
                "mean_abs_shap": round(feat_score, 5),
                "spearman_corr_with_shap": round(corr, 4),
                "direction": dir_label
            }
            target_features_data.append(feat_record)

            feature_importance_rows.append({
                "target": target,
                "rank": rank,
                "feature": feat_name,
                "mean_abs_shap": round(feat_score, 5),
                "spearman_corr_with_shap": round(corr, 4),
                "direction": dir_label
            })

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
        else:
            base_val = float(base_val)

        shap_results_by_target[target] = {
            "model_id": config["model_id"],
            "threshold": config["threshold"],
            "sample_size": SHAP_SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "cohort": "validation",
            "base_value_log_odds": base_val,
            "features": target_features_data
        }

        top5 = [rec["feature"] for rec in target_features_data[:5]]
        top10 = [rec["feature"] for rec in target_features_data[:10]]
        print(f"  Top 5 Features: {', '.join(top5)}")

        # Exploratory interaction analysis on top 5 features using 250 samples
        df_inter_sample = df_val_target.sample(n=INTERACTION_SAMPLE_SIZE, random_state=RANDOM_SEED)
        X_inter_sample = transform_candidate_features_v2(df_inter_sample)
        X_inter_trans = preprocessor.transform(X_inter_sample)
        top_interactions = run_exploratory_interactions(explainer, X_inter_trans, transformed_cols, original_features, top5)

        top_features_summary[target] = {
            "top_5": top5,
            "top_10": top10,
            "top_5_details": target_features_data[:5],
            "top_interactions": top_interactions
        }

        # Generate target figures
        generate_shap_visualizations(
            target=target,
            sv_34=sv_30,
            X_sample=X_sample,
            original_features=original_features,
            mean_abs_shap=mean_abs_shap,
            ranked_indices=ranked_indices,
            suffix="_v2"
        )

    # 4. Cross-Target Synthesis & Comparison Table
    print("\n[*] Constructing Cross-Target Synthesis (v2)...")
    cross_target_rows = []
    for feat in original_features:
        stunt_rec = next(r for r in shap_results_by_target["stunting"]["features"] if r["feature"] == feat)
        under_rec = next(r for r in shap_results_by_target["underweight"]["features"] if r["feature"] == feat)
        waste_rec = next(r for r in shap_results_by_target["wasting"]["features"] if r["feature"] == feat)

        mean_rank = (stunt_rec["rank"] + under_rec["rank"] + waste_rec["rank"]) / 3.0
        cross_target_rows.append({
            "feature": feat,
            "rank_stunting": stunt_rec["rank"],
            "rank_underweight": under_rec["rank"],
            "rank_wasting": waste_rec["rank"],
            "mean_rank_across_targets": round(mean_rank, 2),
            "mean_abs_shap_stunting": stunt_rec["mean_abs_shap"],
            "mean_abs_shap_underweight": under_rec["mean_abs_shap"],
            "mean_abs_shap_wasting": waste_rec["mean_abs_shap"],
            "direction_stunting": stunt_rec["direction"],
            "direction_underweight": under_rec["direction"],
            "direction_wasting": waste_rec["direction"]
        })

    df_cross = pd.DataFrame(cross_target_rows).sort_values("mean_rank_across_targets", ascending=True)
    df_cross.to_csv(OUTPUT_CROSS_TARGET_CSV_V2, index=False)
    print(f"[+] Saved Cross-Target CSV: {OUTPUT_CROSS_TARGET_CSV_V2}")

    # Plot Cross-Target Comparison Figure
    plot_cross_target_comparison(df_cross, suffix="_v2")

    # 5. Export Feature Importance CSV and JSON
    df_importance = pd.DataFrame(feature_importance_rows)
    df_importance.to_csv(OUTPUT_SHAP_CSV_V2, index=False)
    print(f"[+] Saved SHAP Feature Importance CSV: {OUTPUT_SHAP_CSV_V2}")

    output_json_data = {
        "step": "Major Step 6 — Model Explainability & Interpretability using TreeSHAP (30 Predictors v2)",
        "scenario": "Scenario A — Community Pre-Screening",
        "feature_count": 30,
        "feature_version": "v2.0.0",
        "dataset": "NFHS-5 India 2019-21 Children's Recode (KR)",
        "explanation_method": "shap.TreeExplainer (TreeSHAP)",
        "explanation_cohort": "Validation partition (strictly held-out from training)",
        "sample_size_per_target": SHAP_SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "total_candidate_features": len(original_features),
        "results_by_target": shap_results_by_target,
        "cross_target_top_consensus": df_cross.head(10)[["feature", "mean_rank_across_targets"]].to_dict(orient="records")
    }

    with open(OUTPUT_SHAP_JSON_V2, "w") as f:
        json.dump(output_json_data, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved SHAP Feature Importance JSON: {OUTPUT_SHAP_JSON_V2}")

    with open(OUTPUT_TOP_FEATURES_JSON_V2, "w") as f:
        json.dump(top_features_summary, f, indent=2, cls=NumpyEncoder)
    print(f"[+] Saved SHAP Top Features JSON: {OUTPUT_TOP_FEATURES_JSON_V2}")

    elapsed = time.time() - start_time
    print(f"\n[OK] Major Step 6 SHAP analysis (v2) executed successfully in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TreeSHAP explainability analysis")
    parser.add_argument("--v2", action="store_true", help="Run SHAP analysis on 30-feature v2 scenario-a models")
    args = parser.parse_args()

    if args.v2:
        run_shap_analysis_v2()
    else:
        run_shap_analysis()
