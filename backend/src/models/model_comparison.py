"""
Step 10: Model Comparison Pipeline.
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
Scenario A — Community Pre-Screening Setting

Performs a controlled, leakage-safe empirical comparison of 5 model families:
1. Logistic Regression (Step 9 Reference Baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

Across 2 class imbalance conditions:
- Experiment A: Unweighted (Standard distribution)
- Experiment B: Class-Weighted (Imbalance-aware training)

Separately for the 3 target outcomes:
- Stunting
- Underweight
- Wasting

Evaluations are conducted strictly on the Validation set at threshold 0.5.
The Test set remains completely locked and unaccessed.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from src.features.build_features import (
    transform_candidate_features,
    transform_candidate_features_v2,
    validate_scenario_a_v2_feature_matrix
)

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745
INPUT_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
FEATURE_METADATA_PATH = "data/interim/feature_metadata.json"
FEATURE_METADATA_V2_PATH = "data/interim/feature_metadata_v2.json"
OUTPUT_METRICS_JSON_PATH = "data/interim/model_comparison_metrics.json"
OUTPUT_METRICS_CSV_PATH = "data/interim/model_comparison_metrics.csv"
OUTPUT_METRICS_V2_JSON_PATH = "data/interim/model_comparison_metrics_v2.json"
OUTPUT_METRICS_V2_CSV_PATH = "data/interim/model_comparison_metrics_v2.csv"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"
MODELS_V2_DIR = "models/v2"

RANDOM_SEED = 42
TARGET_NAMES = ["stunting", "underweight", "wasting"]
DEFAULT_THRESHOLD = 0.5


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


def build_preprocessing_pipeline(feature_names: list) -> tuple:
    """
    Constructs a leakage-safe ColumnTransformer for the 34 candidate features.
    """
    cat_cols = [
        "child_age_group",
        "delivery_place_type",
        "caste_category",
        "religion_category",
        "drinking_water_type",
        "sanitation_facility_type",
        "state_id"
    ]
    num_cols = [c for c in feature_names if c not in cat_cols]
    assert len(cat_cols) + len(num_cols) == 34, f"Feature count mismatch: {len(cat_cols)} + {len(num_cols)} != 34"

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols


def build_preprocessing_pipeline_v2(feature_names: list) -> tuple:
    """
    Constructs a leakage-safe ColumnTransformer for the 30 candidate features.
    """
    cat_cols = [
        "child_age_group",
        "delivery_place_type",
        "drinking_water_type",
        "sanitation_facility_type",
        "state_id"
    ]
    num_cols = [c for c in feature_names if c not in cat_cols]
    assert len(cat_cols) + len(num_cols) == 30, f"Feature count mismatch: {len(cat_cols)} + {len(num_cols)} != 30"

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols


def get_model_instance(model_name: str, weighted: bool, pos_neg_ratio: float = 1.0):
    """
    Instantiates the model with documented, conservative default hyperparameters.
    """
    if model_name == "Logistic Regression":
        cw = "balanced" if weighted else None
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight=cw,
            random_state=RANDOM_SEED
        )
    elif model_name == "Random Forest":
        cw = "balanced" if weighted else None
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            class_weight=cw,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif model_name == "XGBoost":
        scale_weight = pos_neg_ratio if weighted else 1.0
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1
        )
    elif model_name == "LightGBM":
        cw = "balanced" if weighted else None
        return LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=cw,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1
        )
    elif model_name == "CatBoost":
        auto_cw = "Balanced" if weighted else None
        return CatBoostClassifier(
            iterations=150,
            depth=6,
            learning_rate=0.08,
            auto_class_weights=auto_cw,
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_experiments(df_raw: pd.DataFrame, X_all: pd.DataFrame) -> list:
    """
    Executes the 30 experiments (5 models x 2 conditions x 3 targets).
    """
    feature_names = list(X_all.columns)
    model_families = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]
    conditions = ["unweighted", "class_weighted"]

    results = []

    # Dictionary to store reference baseline metrics for calculating deltas
    baseline_reference = {}

    for target_name in TARGET_NAMES:
        print(f"\n{'='*30} TARGET: {target_name.upper()} {'='*30}")
        elig_col = f"eligible_{target_name}"
        valid_mask = df_raw[elig_col] == True

        train_mask = (df_raw["split"] == "train") & valid_mask
        val_mask = (df_raw["split"] == "val") & valid_mask
        test_mask = (df_raw["split"] == "test") & valid_mask

        # Strict partition assertions
        assert (train_mask & val_mask).sum() == 0, "Train and Validation masks overlap!"
        assert (train_mask & test_mask).sum() == 0, "Train and Test masks overlap!"
        assert (val_mask & test_mask).sum() == 0, "Validation and Test masks overlap!"

        X_train = X_all.loc[train_mask].copy()
        y_train = df_raw.loc[train_mask, target_name].astype(int)

        X_val = X_all.loc[val_mask].copy()
        y_val = df_raw.loc[val_mask, target_name].astype(int)

        n_train = len(X_train)
        n_val = len(X_val)
        train_pos = int((y_train == 1).sum())
        train_neg = int((y_train == 0).sum())
        pos_neg_ratio = round(train_neg / train_pos, 4) if train_pos > 0 else 1.0

        val_pos = int((y_val == 1).sum())
        val_neg = int((y_val == 0).sum())
        val_prev = round((val_pos / n_val) * 100, 2)

        print(f"Cohort Sizes: Train N={n_train:,} (Pos: {train_pos:,}, Neg: {train_neg:,}, Ratio: {pos_neg_ratio:.3f}) | Val N={n_val:,} (Prev: {val_prev:.2f}%)")

        for model_name in model_families:
            for cond in conditions:
                is_weighted = (cond == "class_weighted")
                print(f"  Training {model_name:<20} | Condition: {cond:<15} ...", end="", flush=True)

                t_start = time.time()
                preprocessor, _, _ = build_preprocessing_pipeline(feature_names)
                clf = get_model_instance(model_name, is_weighted, pos_neg_ratio)

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])

                # Fit strictly on training partition
                pipeline.fit(X_train, y_train)
                train_time = round(time.time() - t_start, 2)

                # Predict on validation partition
                y_val_prob = pipeline.predict_proba(X_val)[:, 1]
                y_val_pred = (y_val_prob >= DEFAULT_THRESHOLD).astype(int)

                # Evaluation metrics
                roc_auc = round(float(roc_auc_score(y_val, y_val_prob)), 4)
                pr_auc = round(float(average_precision_score(y_val, y_val_prob)), 4)
                accuracy = round(float(accuracy_score(y_val, y_val_pred)), 4)
                precision = round(float(precision_score(y_val, y_val_pred, zero_division=0)), 4)
                recall = round(float(recall_score(y_val, y_val_pred, zero_division=0)), 4)
                f1 = round(float(f1_score(y_val, y_val_pred, zero_division=0)), 4)
                cm = confusion_matrix(y_val, y_val_pred).tolist()

                # Store baseline reference for LR unweighted
                if model_name == "Logistic Regression" and cond == "unweighted":
                    baseline_reference[target_name] = {
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "recall": recall,
                        "f1": f1
                    }

                base_ref = baseline_reference.get(target_name, {"roc_auc": roc_auc, "pr_auc": pr_auc, "recall": recall, "f1": f1})
                delta_roc = round(roc_auc - base_ref["roc_auc"], 4)
                delta_pr = round(pr_auc - base_ref["pr_auc"], 4)
                delta_recall = round(recall - base_ref["recall"], 4)
                delta_f1 = round(f1 - base_ref["f1"], 4)

                print(f" Done ({train_time}s) -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

                # Save pipeline artifact for key configurations
                model_slug = model_name.lower().replace(" ", "_")
                artifact_filename = f"model_comparison_{model_slug}_{target_name}_{cond}.joblib"
                artifact_path = os.path.join(MODELS_DIR, artifact_filename)
                os.makedirs(MODELS_DIR, exist_ok=True)
                joblib.dump(pipeline, artifact_path)

                record = {
                    "target": target_name,
                    "model": model_name,
                    "condition": cond,
                    "train_n": n_train,
                    "val_n": n_val,
                    "val_prevalence_pct": val_prev,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "delta_roc_auc": delta_roc,
                    "delta_pr_auc": delta_pr,
                    "delta_recall": delta_recall,
                    "delta_f1": delta_f1,
                    "train_time_sec": train_time,
                    "confusion_matrix": cm,
                    "artifact_file": artifact_filename
                }
                results.append(record)

    return results


def run_hyperparameter_exploration(df_raw: pd.DataFrame, X_all: pd.DataFrame) -> dict:
    """
    Conducts a controlled, modest hyperparameter exploration for LightGBM on Stunting.
    Explores tree depth, leaves, and learning rate strictly on training data evaluated on validation.
    """
    print("\n" + "=" * 70)
    print("Controlled Hyperparameter Exploration (LightGBM on Stunting)")
    print("=" * 70)

    target_name = "stunting"
    valid_mask = df_raw[f"eligible_{target_name}"] == True
    train_mask = (df_raw["split"] == "train") & valid_mask
    val_mask = (df_raw["split"] == "val") & valid_mask

    X_train = X_all.loc[train_mask]
    y_train = df_raw.loc[train_mask, target_name].astype(int)
    X_val = X_all.loc[val_mask]
    y_val = df_raw.loc[val_mask, target_name].astype(int)

    candidates = [
        {"name": "Conservative Shallow", "n_estimators": 100, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.05},
        {"name": "Step 10 Default", "n_estimators": 100, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.08},
        {"name": "Higher Capacity", "n_estimators": 150, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.08},
        {"name": "Low Learning Rate Extended", "n_estimators": 200, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.03}
    ]

    hp_results = []
    preprocessor, _, _ = build_preprocessing_pipeline(list(X_all.columns))

    for cfg in candidates:
        name = cfg.pop("name")
        clf = LGBMClassifier(**cfg, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc = round(float(roc_auc_score(y_val, y_prob)), 4)
        pr = round(float(average_precision_score(y_val, y_prob)), 4)
        acc = round(float(accuracy_score(y_val, y_pred)), 4)
        f1 = round(float(f1_score(y_val, y_pred, zero_division=0)), 4)

        print(f"  Configuration: {name:<28} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        cfg["name"] = name
        cfg["roc_auc"] = roc
        cfg["pr_auc"] = pr
        cfg["accuracy"] = acc
        cfg["f1_score"] = f1
        hp_results.append(cfg)

    return {
        "target": "stunting",
        "model": "LightGBM",
        "search_space_description": "Controlled exploration across depth (4-8), leaves (15-63), and lr (0.03-0.08)",
        "evaluations": hp_results
    }


def generate_visualizations(results_df: pd.DataFrame):
    """
    Generates publication-quality comparison charts saved to reports/figures/.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("muted")

    # 1. ROC-AUC Comparison (Fig 8)
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        data=results_df,
        x="target",
        y="roc_auc",
        hue="model",
        palette="tab10"
    )
    plt.title("Figure 8: Validation ROC-AUC Across 5 Model Families and 3 Nutritional Targets", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Nutritional Target", fontsize=12, labelpad=10)
    plt.ylabel("Validation ROC-AUC", fontsize=12)
    plt.ylim(0.55, 0.75)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance Baseline (0.50)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model Family")
    plt.tight_layout()
    fig8_path = os.path.join(FIGURES_DIR, "fig8_model_comparison_roc_auc.png")
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig8_path}")

    # 2. PR-AUC Comparison (Fig 9)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df,
        x="target",
        y="pr_auc",
        hue="model",
        palette="tab10"
    )
    plt.title("Figure 9: Validation PR-AUC (Average Precision) Across Models and Targets", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Nutritional Target", fontsize=12, labelpad=10)
    plt.ylabel("Validation PR-AUC", fontsize=12)
    plt.ylim(0.15, 0.60)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model Family")
    plt.tight_layout()
    fig9_path = os.path.join(FIGURES_DIR, "fig9_model_comparison_pr_auc.png")
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig9_path}")

    # 3. Class Weighting Effect on Recall and Precision (Fig 10)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df,
        x="model",
        y="recall",
        hue="condition",
        palette=["#4C72B0", "#DD8452"]
    )
    plt.title("Figure 10: Effect of Class Weighting on Clinical Sensitivity (Recall) Across Models", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model Family", fontsize=12, labelpad=10)
    plt.ylabel("Validation Recall (Sensitivity at Threshold 0.5)", fontsize=12)
    plt.ylim(0.0, 0.85)
    plt.legend(title="Condition", labels=["Unweighted (Standard)", "Class-Weighted (Balanced)"])
    plt.tight_layout()
    fig10_path = os.path.join(FIGURES_DIR, "fig10_class_weighting_effect.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig10_path}")

    # 4. F1-Score Trade-off Across Targets (Fig 11)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df,
        x="target",
        y="f1_score",
        hue="condition",
        palette=["#55A868", "#C44E52"]
    )
    plt.title("Figure 11: F1-Score Trade-offs Between Unweighted and Class-Weighted Models", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Nutritional Target", fontsize=12, labelpad=10)
    plt.ylabel("Validation F1-Score", fontsize=12)
    plt.ylim(0.0, 0.60)
    plt.legend(title="Condition", labels=["Unweighted", "Class-Weighted"])
    plt.tight_layout()
    fig11_path = os.path.join(FIGURES_DIR, "fig11_model_comparison_f1_tradeoff.png")
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig11_path}")


def run_experiments_v2(df_raw: pd.DataFrame, X_all: pd.DataFrame) -> list:
    """
    Executes the 30 experiments (5 models x 2 conditions x 3 targets) on 30 features.
    Saves artifacts to models/v2/.
    """
    feature_names = list(X_all.columns)
    model_families = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]
    conditions = ["unweighted", "class_weighted"]

    results = []
    baseline_reference = {}

    for target_name in TARGET_NAMES:
        print(f"\n{'='*30} TARGET: {target_name.upper()} (30 Features v2) {'='*30}")
        elig_col = f"eligible_{target_name}"
        valid_mask = df_raw[elig_col] == True

        train_mask = (df_raw["split"] == "train") & valid_mask
        val_mask = (df_raw["split"] == "val") & valid_mask
        test_mask = (df_raw["split"] == "test") & valid_mask

        assert (train_mask & val_mask).sum() == 0, "Train and Validation masks overlap!"
        assert (train_mask & test_mask).sum() == 0, "Train and Test masks overlap!"
        assert (val_mask & test_mask).sum() == 0, "Validation and Test masks overlap!"

        X_train = X_all.loc[train_mask].copy()
        y_train = df_raw.loc[train_mask, target_name].astype(int)

        X_val = X_all.loc[val_mask].copy()
        y_val = df_raw.loc[val_mask, target_name].astype(int)

        n_train = len(X_train)
        n_val = len(X_val)
        train_pos = int((y_train == 1).sum())
        train_neg = int((y_train == 0).sum())
        pos_neg_ratio = round(train_neg / train_pos, 4) if train_pos > 0 else 1.0

        val_pos = int((y_val == 1).sum())
        val_neg = int((y_val == 0).sum())
        val_prev = round((val_pos / n_val) * 100, 2)

        print(f"Cohort Sizes: Train N={n_train:,} (Pos: {train_pos:,}, Neg: {train_neg:,}, Ratio: {pos_neg_ratio:.3f}) | Val N={n_val:,} (Prev: {val_prev:.2f}%)")

        for model_name in model_families:
            for cond in conditions:
                is_weighted = (cond == "class_weighted")
                print(f"  Training {model_name:<20} | Condition: {cond:<15} ...", end="", flush=True)

                t_start = time.time()
                preprocessor, _, _ = build_preprocessing_pipeline_v2(feature_names)
                clf = get_model_instance(model_name, is_weighted, pos_neg_ratio)

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", clf)
                ])

                pipeline.fit(X_train, y_train)
                train_time = round(time.time() - t_start, 2)

                y_val_prob = pipeline.predict_proba(X_val)[:, 1]
                y_val_pred = (y_val_prob >= DEFAULT_THRESHOLD).astype(int)

                roc_auc = round(float(roc_auc_score(y_val, y_val_prob)), 4)
                pr_auc = round(float(average_precision_score(y_val, y_val_prob)), 4)
                accuracy = round(float(accuracy_score(y_val, y_val_pred)), 4)
                precision = round(float(precision_score(y_val, y_val_pred, zero_division=0)), 4)
                recall = round(float(recall_score(y_val, y_val_pred, zero_division=0)), 4)
                f1 = round(float(f1_score(y_val, y_val_pred, zero_division=0)), 4)
                cm = confusion_matrix(y_val, y_val_pred).tolist()

                if model_name == "Logistic Regression" and cond == "unweighted":
                    baseline_reference[target_name] = {
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "recall": recall,
                        "f1": f1
                    }

                base_ref = baseline_reference.get(target_name, {"roc_auc": roc_auc, "pr_auc": pr_auc, "recall": recall, "f1": f1})
                delta_roc = round(roc_auc - base_ref["roc_auc"], 4)
                delta_pr = round(pr_auc - base_ref["pr_auc"], 4)
                delta_recall = round(recall - base_ref["recall"], 4)
                delta_f1 = round(f1 - base_ref["f1"], 4)

                print(f" Done ({train_time}s) -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

                model_slug = model_name.lower().replace(" ", "_")
                artifact_filename = f"model_comparison_{model_slug}_{target_name}_{cond}_v2.joblib"
                artifact_path = os.path.join(MODELS_V2_DIR, artifact_filename)
                os.makedirs(MODELS_V2_DIR, exist_ok=True)
                joblib.dump(pipeline, artifact_path)

                record = {
                    "target": target_name,
                    "model": model_name,
                    "condition": cond,
                    "train_n": n_train,
                    "val_n": n_val,
                    "val_prevalence_pct": val_prev,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "delta_roc_auc": delta_roc,
                    "delta_pr_auc": delta_pr,
                    "delta_recall": delta_recall,
                    "delta_f1": delta_f1,
                    "train_time_sec": train_time,
                    "confusion_matrix": cm,
                    "artifact_file": artifact_filename
                }
                results.append(record)

    return results


def run_hyperparameter_exploration_v2(df_raw: pd.DataFrame, X_all: pd.DataFrame) -> dict:
    """
    Controlled hyperparameter exploration for LightGBM on Stunting using 30 features.
    """
    print("\n" + "=" * 70)
    print("Controlled Hyperparameter Exploration (LightGBM on Stunting, 30 Features v2)")
    print("=" * 70)

    target_name = "stunting"
    valid_mask = df_raw[f"eligible_{target_name}"] == True
    train_mask = (df_raw["split"] == "train") & valid_mask
    val_mask = (df_raw["split"] == "val") & valid_mask

    X_train = X_all.loc[train_mask]
    y_train = df_raw.loc[train_mask, target_name].astype(int)
    X_val = X_all.loc[val_mask]
    y_val = df_raw.loc[val_mask, target_name].astype(int)

    candidates = [
        {"name": "Conservative Shallow", "n_estimators": 100, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.05},
        {"name": "Default 30-feat", "n_estimators": 100, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.08},
        {"name": "Higher Capacity", "n_estimators": 150, "max_depth": 8, "num_leaves": 63, "learning_rate": 0.08},
        {"name": "Low Learning Rate Extended", "n_estimators": 200, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.03}
    ]

    hp_results = []
    preprocessor, _, _ = build_preprocessing_pipeline_v2(list(X_all.columns))

    for cfg in candidates:
        name = cfg.pop("name")
        clf = LGBMClassifier(**cfg, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc = round(float(roc_auc_score(y_val, y_prob)), 4)
        pr = round(float(average_precision_score(y_val, y_prob)), 4)
        acc = round(float(accuracy_score(y_val, y_pred)), 4)
        f1 = round(float(f1_score(y_val, y_pred, zero_division=0)), 4)

        print(f"  Configuration: {name:<28} | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        cfg["name"] = name
        cfg["roc_auc"] = roc
        cfg["pr_auc"] = pr
        cfg["accuracy"] = acc
        cfg["f1_score"] = f1
        hp_results.append(cfg)

    return {
        "target": "stunting",
        "model": "LightGBM",
        "search_space_description": "Controlled exploration across depth (4-8), leaves (15-63), and lr (0.03-0.08) on 30 features",
        "evaluations": hp_results
    }


def run_model_comparison_v2():
    print("=" * 70)
    print("NutriSense AI — Model Comparison Pipeline (30-Feature Scenario-A v2)")
    print("=" * 70)

    # 1. Immutability audit
    verify_raw_dataset_immutability()

    # 2. Load dataset and v2 feature registry
    print(f"\nLoading cleaned data from: {INPUT_DATA_PATH}")
    df_raw = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
    assert "split" in df_raw.columns, "Missing 'split' column! Step 8 required."

    with open(FEATURE_METADATA_V2_PATH, "r") as f:
        feature_meta_v2 = json.load(f)
    approved_features = feature_meta_v2["features"]
    assert len(approved_features) == 30, f"Expected 30 features, found {len(approved_features)}"
    print(f"[OK] Verified 30 approved Scenario A candidate features from v2 metadata.")

    # 3. Transform candidate features v2
    print("Constructing 30-feature matrix X...")
    X_all = transform_candidate_features_v2(df_raw)
    validate_scenario_a_v2_feature_matrix(X_all)

    # 4. Execute 30 controlled experiments
    results = run_experiments_v2(df_raw, X_all)

    # 5. Execute controlled hyperparameter exploration
    hp_exploration = run_hyperparameter_exploration_v2(df_raw, X_all)

    # 6. Save results to CSV and JSON
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_METRICS_V2_CSV_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_METRICS_V2_CSV_PATH, index=False)
    print(f"\n[OK] Model comparison metrics v2 CSV saved: {OUTPUT_METRICS_V2_CSV_PATH}")

    metadata_output = {
        "step": "Model Comparison (30-Feature Scenario-A)",
        "scenario": "Scenario A — Community Pre-Screening",
        "feature_count": 30,
        "random_state": RANDOM_SEED,
        "models_evaluated": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
        "imbalance_conditions": ["unweighted", "class_weighted"],
        "targets": TARGET_NAMES,
        "test_set_status": {
            "locked": True,
            "accessed": False,
            "statement": "The test set was not used for model comparison, fitting, weighting selection, or threshold tuning."
        },
        "total_experiments_conducted": len(results),
        "results": results,
        "hyperparameter_exploration": hp_exploration
    }

    with open(OUTPUT_METRICS_V2_JSON_PATH, "w") as f:
        json.dump(metadata_output, f, indent=2)
    print(f"[OK] Model comparison metrics v2 JSON saved: {OUTPUT_METRICS_V2_JSON_PATH}")

    print("\n" + "=" * 70)
    print("Model Comparison (30-feature v2) completed successfully!")
    print("=" * 70)
    return metadata_output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model Comparison for Scenario A.")
    parser.add_argument("--v2", action="store_true", help="Run 30-feature v2 model comparison experiments.")
    parser.add_argument("--all", action="store_true", help="Run both v1 and v2 model comparison experiments.")
    args = parser.parse_args()

    if args.v2:
        run_model_comparison_v2()
        return

    print("=" * 70)
    print("NutriSense AI — Step 10: Model Comparison Pipeline")
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
    assert len(approved_features) == 34, f"Expected 34 features, found {len(approved_features)}"
    print(f"[OK] Verified 34 approved Scenario A candidate features.")

    # 3. Transform candidate features
    print("Constructing 34-feature matrix X...")
    X_all = transform_candidate_features(df_raw)
    assert list(X_all.columns) == approved_features

    # 4. Execute 30 controlled experiments
    results = run_experiments(df_raw, X_all)

    # 5. Execute controlled hyperparameter exploration
    hp_exploration = run_hyperparameter_exploration(df_raw, X_all)

    # 6. Save results to CSV and JSON
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_METRICS_CSV_PATH, index=False)
    print(f"\n[OK] Model comparison metrics CSV saved: {OUTPUT_METRICS_CSV_PATH}")

    metadata_output = {
        "step": "Step 10 — Model Comparison",
        "scenario": "Scenario A — Community Pre-Screening",
        "random_state": RANDOM_SEED,
        "models_evaluated": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
        "imbalance_conditions": ["unweighted", "class_weighted"],
        "targets": TARGET_NAMES,
        "test_set_status": {
            "locked": True,
            "accessed": False,
            "statement": "The test set was not used for model comparison, fitting, weighting selection, or threshold tuning."
        },
        "total_experiments_conducted": len(results),
        "results": results,
        "hyperparameter_exploration": hp_exploration
    }

    with open(OUTPUT_METRICS_JSON_PATH, "w") as f:
        json.dump(metadata_output, f, indent=2)
    print(f"[OK] Model comparison metrics JSON saved: {OUTPUT_METRICS_JSON_PATH}")

    # 7. Generate visualization figures
    print("\nGenerating comparison visualizations...")
    generate_visualizations(results_df)

    if args.all:
        run_model_comparison_v2()

    print("\n" + "=" * 70)
    print("Step 10: Model Comparison completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
