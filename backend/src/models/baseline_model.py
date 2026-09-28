"""
Step 9: Baseline Model Training and Validation Pipeline.
NutriSense AI: Multimodal Childhood Malnutrition Risk Intelligence and Early Intervention System
Scenario A — Community Pre-Screening Setting

Trains interpretable Logistic Regression baseline models for:
1. Stunting
2. Underweight
3. Wasting

Enforces strict leakage-safe principles:
- Preprocessing (imputation, scaling, one-hot encoding) is encapsulated in an sklearn Pipeline.
- Pipeline is fitted strictly on the Training partition (X_train).
- Evaluation is performed exclusively on the Validation partition (X_val) at threshold 0.5.
- The Test set is strictly locked and never accessed during fitting, tuning, or evaluation.
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

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
OUTPUT_METRICS_PATH = "data/interim/baseline_metrics.json"
OUTPUT_METRICS_V2_PATH = "data/interim/baseline_metrics_v2.json"
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
    Returns:
        preprocessor: ColumnTransformer instance
        num_cols: list of numerical/binary feature names
        cat_cols: list of categorical/nominal feature names
    """
    # 7 categorical/nominal features requiring one-hot encoding
    cat_cols = [
        "child_age_group",
        "delivery_place_type",
        "caste_category",
        "religion_category",
        "drinking_water_type",
        "sanitation_facility_type",
        "state_id"
    ]
    # 27 numerical, ordinal, and binary features
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


def train_and_evaluate_target(
    target_name: str,
    df_raw: pd.DataFrame,
    X_all: pd.DataFrame,
    num_cols: list,
    cat_cols: list
) -> tuple:
    """
    Trains Logistic Regression on X_train and evaluates on X_val for a specific target.
    """
    elig_col = f"eligible_{target_name}"
    valid_mask = df_raw[elig_col] == True

    train_mask = (df_raw["split"] == "train") & valid_mask
    val_mask = (df_raw["split"] == "val") & valid_mask
    test_mask = (df_raw["split"] == "test") & valid_mask

    # Strict partition isolation assertions
    assert (train_mask & val_mask).sum() == 0, "Train and Validation masks overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train and Test masks overlap!"
    assert (val_mask & test_mask).sum() == 0, "Validation and Test masks overlap!"

    X_train = X_all.loc[train_mask].copy()
    y_train = df_raw.loc[train_mask, target_name].astype(int)

    X_val = X_all.loc[val_mask].copy()
    y_val = df_raw.loc[val_mask, target_name].astype(int)

    n_train = len(X_train)
    n_val = len(X_val)
    val_pos = int((y_val == 1).sum())
    val_neg = int((y_val == 0).sum())
    val_prev = round((val_pos / n_val) * 100, 2)

    print(f"\n[{target_name.upper()}] Training baseline model...")
    print(f"  Training cohort:   N = {n_train:,} (Pos: {(y_train == 1).sum():,}, Neg: {(y_train == 0).sum():,})")
    print(f"  Validation cohort: N = {n_val:,} (Pos: {val_pos:,}, Neg: {val_neg:,}, Prev: {val_prev:.2f}%)")

    # Build Pipeline
    preprocessor, _, _ = build_preprocessing_pipeline(list(X_all.columns))
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Fit exclusively on training data
    pipeline.fit(X_train, y_train)

    # Post-preprocessing feature count
    prep_fitted = pipeline.named_steps["preprocessor"]
    n_features_after_prep = prep_fitted.transform(X_train.iloc[:2]).shape[1]

    # Predict on validation data
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= DEFAULT_THRESHOLD).astype(int)

    # Calculate metrics
    roc_auc = round(float(roc_auc_score(y_val, y_val_prob)), 4)
    pr_auc = round(float(average_precision_score(y_val, y_val_prob)), 4)
    accuracy = round(float(accuracy_score(y_val, y_val_pred)), 4)
    precision = round(float(precision_score(y_val, y_val_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_val, y_val_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_val, y_val_pred, zero_division=0)), 4)
    cm = confusion_matrix(y_val, y_val_pred).tolist()

    print(f"  Validation Results (Threshold = {DEFAULT_THRESHOLD}):")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    PR-AUC:    {pr_auc:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")

    # Save model artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"baseline_logistic_{target_name}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"  [OK] Saved model artifact: {model_path}")

    metrics_dict = {
        "model_name": "Logistic Regression Baseline",
        "target": target_name,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "threshold": DEFAULT_THRESHOLD,
        "train_sample_count": n_train,
        "train_positive_count": int((y_train == 1).sum()),
        "train_negative_count": int((y_train == 0).sum()),
        "validation_sample_count": n_val,
        "validation_positive_count": val_pos,
        "validation_negative_count": val_neg,
        "validation_prevalence_pct": val_prev,
        "features_before_preprocessing": len(X_all.columns),
        "features_after_preprocessing": int(n_features_after_prep),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm
        }
    }

    return pipeline, metrics_dict


def build_preprocessing_pipeline_v2(feature_names: list) -> tuple:
    """
    Constructs a leakage-safe ColumnTransformer for the 30 Scenario A v2 candidate features.
    Returns:
        preprocessor: ColumnTransformer instance
        num_cols: list of 25 numerical/binary/ordinal feature names
        cat_cols: list of 5 categorical/nominal feature names
    """
    cat_cols = [
        "child_age_group",
        "delivery_place_type",
        "drinking_water_type",
        "sanitation_facility_type",
        "state_id"
    ]
    num_cols = [c for c in feature_names if c not in cat_cols]
    assert len(cat_cols) + len(num_cols) == 30, f"v2 feature count mismatch: {len(cat_cols)} + {len(num_cols)} != 30"

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


def train_and_evaluate_target_v2(
    target_name: str,
    df_raw: pd.DataFrame,
    X_all: pd.DataFrame,
    num_cols: list,
    cat_cols: list
) -> tuple:
    """
    Trains Logistic Regression on X_train and evaluates on X_val for a specific target on 30 features.
    Saves artifact in models/v2/baseline_logistic_{target_name}_v2.joblib.
    """
    elig_col = f"eligible_{target_name}"
    valid_mask = df_raw[elig_col] == True

    train_mask = (df_raw["split"] == "train") & valid_mask
    val_mask = (df_raw["split"] == "val") & valid_mask
    test_mask = (df_raw["split"] == "test") & valid_mask

    # Strict partition isolation assertions
    assert (train_mask & val_mask).sum() == 0, "Train and Validation masks overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train and Test masks overlap!"
    assert (val_mask & test_mask).sum() == 0, "Validation and Test masks overlap!"

    X_train = X_all.loc[train_mask].copy()
    y_train = df_raw.loc[train_mask, target_name].astype(int)

    X_val = X_all.loc[val_mask].copy()
    y_val = df_raw.loc[val_mask, target_name].astype(int)

    n_train = len(X_train)
    n_val = len(X_val)
    val_pos = int((y_val == 1).sum())
    val_neg = int((y_val == 0).sum())
    val_prev = round((val_pos / n_val) * 100, 2)

    print(f"\n[{target_name.upper()} v2] Training baseline model (30 features)...")
    print(f"  Training cohort:   N = {n_train:,} (Pos: {(y_train == 1).sum():,}, Neg: {(y_train == 0).sum():,})")
    print(f"  Validation cohort: N = {n_val:,} (Pos: {val_pos:,}, Neg: {val_neg:,}, Prev: {val_prev:.2f}%)")

    # Build Pipeline
    preprocessor, _, _ = build_preprocessing_pipeline_v2(list(X_all.columns))
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Fit exclusively on training data
    pipeline.fit(X_train, y_train)

    # Post-preprocessing feature count
    prep_fitted = pipeline.named_steps["preprocessor"]
    n_features_after_prep = prep_fitted.transform(X_train.iloc[:2]).shape[1]

    # Predict on validation data
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= DEFAULT_THRESHOLD).astype(int)

    # Calculate metrics
    roc_auc = round(float(roc_auc_score(y_val, y_val_prob)), 4)
    pr_auc = round(float(average_precision_score(y_val, y_val_prob)), 4)
    accuracy = round(float(accuracy_score(y_val, y_val_pred)), 4)
    precision = round(float(precision_score(y_val, y_val_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_val, y_val_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_val, y_val_pred, zero_division=0)), 4)
    cm = confusion_matrix(y_val, y_val_pred).tolist()

    print(f"  Validation Results (Threshold = {DEFAULT_THRESHOLD}):")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    PR-AUC:    {pr_auc:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")

    # Save model artifact in models/v2/
    os.makedirs(MODELS_V2_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_V2_DIR, f"baseline_logistic_{target_name}_v2.joblib")
    joblib.dump(pipeline, model_path)
    print(f"  [OK] Saved model artifact: {model_path}")

    metrics_dict = {
        "model_name": "Logistic Regression Baseline (30-Feature)",
        "target": target_name,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "threshold": DEFAULT_THRESHOLD,
        "train_sample_count": n_train,
        "train_positive_count": int((y_train == 1).sum()),
        "train_negative_count": int((y_train == 0).sum()),
        "validation_sample_count": n_val,
        "validation_positive_count": val_pos,
        "validation_negative_count": val_neg,
        "validation_prevalence_pct": val_prev,
        "features_before_preprocessing": len(X_all.columns),
        "features_after_preprocessing": int(n_features_after_prep),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm
        }
    }

    return pipeline, metrics_dict


def run_baseline_v2():
    print("=" * 70)
    print("NutriSense AI — Baseline Model (Logistic Regression 30-Feature v2)")
    print("=" * 70)

    # 1. Verify raw data immutability
    verify_raw_dataset_immutability()

    # 2. Load cleaned data with targets and split
    print(f"\nLoading interim dataset from: {INPUT_DATA_PATH}")
    df_raw = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
    assert "split" in df_raw.columns, "Missing 'split' column! Ensure Step 8 was completed."

    # 3. Load v2 feature registry
    with open(FEATURE_METADATA_V2_PATH, "r") as f:
        feature_meta_v2 = json.load(f)
    approved_features = feature_meta_v2["features"]
    assert len(approved_features) == 30, f"Expected 30 features, found {len(approved_features)}"
    print(f"[OK] Verified approved 30 Scenario A candidate features from feature_metadata_v2.json.")

    # 4. Extract 30 candidate features
    print("Extracting 30 Scenario A candidate features...")
    X_all = transform_candidate_features_v2(df_raw)
    validate_scenario_a_v2_feature_matrix(X_all)
    print(f"[OK] Feature matrix constructed: {X_all.shape}")

    # 5. Build preprocessing pipeline specifications
    _, num_cols, cat_cols = build_preprocessing_pipeline_v2(approved_features)
    print(f"[OK] Preprocessing architecture: {len(num_cols)} numerical/binary + {len(cat_cols)} categorical features.")

    # 6. Train and evaluate baseline for all 3 targets
    all_metrics = {
        "step": "Baseline Model (30-Feature Scenario-A)",
        "scenario": "Scenario A — Community Pre-Screening",
        "random_state": RANDOM_SEED,
        "model_family": "LogisticRegression",
        "feature_count": 30,
        "preprocessing": {
            "numerical_imputer": "SimpleImputer(strategy='median')",
            "numerical_scaler": "StandardScaler()",
            "categorical_imputer": "SimpleImputer(strategy='most_frequent')",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "leakage_isolation": "Fitted strictly on training partition (X_train) inside sklearn Pipeline"
        },
        "test_set_status": {
            "locked": True,
            "accessed": False,
            "statement": (
                "The test set was not used for baseline model fitting, preprocessing, "
                "model selection, threshold selection, or performance evaluation."
            )
        },
        "targets": {}
    }

    for target_name in TARGET_NAMES:
        _, target_metrics = train_and_evaluate_target_v2(
            target_name=target_name,
            df_raw=df_raw,
            X_all=X_all,
            num_cols=num_cols,
            cat_cols=cat_cols
        )
        all_metrics["targets"][target_name] = target_metrics

    # 7. Save baseline metrics v2 JSON
    os.makedirs(os.path.dirname(OUTPUT_METRICS_V2_PATH), exist_ok=True)
    with open(OUTPUT_METRICS_V2_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] Baseline metrics v2 successfully saved to: {OUTPUT_METRICS_V2_PATH}")

    print("\n" + "=" * 70)
    print("Baseline model training and validation (30-feature v2) completed successfully!")
    print("=" * 70)
    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Logistic Regression for Scenario A.")
    parser.add_argument("--v2", action="store_true", help="Train 30-feature v2 baseline models.")
    parser.add_argument("--all", action="store_true", help="Train both v1 and v2 baseline models.")
    args = parser.parse_args()

    if args.v2:
        run_baseline_v2()
        return

    print("=" * 70)
    print("NutriSense AI — Step 9: Baseline Model (Logistic Regression)")
    print("=" * 70)

    # 1. Verify raw data immutability
    verify_raw_dataset_immutability()

    # 2. Load cleaned data with targets and split
    print(f"\nLoading interim dataset from: {INPUT_DATA_PATH}")
    df_raw = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
    assert "split" in df_raw.columns, "Missing 'split' column! Ensure Step 8 was completed."

    # 3. Load feature registry
    with open(FEATURE_METADATA_PATH, "r") as f:
        feature_meta = json.load(f)
    approved_features = list(feature_meta["feature_statistics"].keys())
    assert len(approved_features) == 34, f"Expected 34 features, found {len(approved_features)}"
    print(f"[OK] Verified approved 34 Scenario A candidate features from Step 7.")

    # 4. Extract 34 candidate features
    print("Extracting 34 Scenario A candidate features...")
    X_all = transform_candidate_features(df_raw)
    assert list(X_all.columns) == approved_features, "Transformed feature columns mismatch approved registry!"
    print(f"[OK] Feature matrix constructed: {X_all.shape}")

    # 5. Build preprocessing pipeline specifications
    _, num_cols, cat_cols = build_preprocessing_pipeline(approved_features)
    print(f"[OK] Preprocessing architecture: {len(num_cols)} numerical/binary + {len(cat_cols)} categorical features.")

    # 6. Train and evaluate baseline for all 3 targets
    all_metrics = {
        "step": "Step 9 — Baseline Model",
        "scenario": "Scenario A — Community Pre-Screening",
        "random_state": RANDOM_SEED,
        "model_family": "LogisticRegression",
        "preprocessing": {
            "numerical_imputer": "SimpleImputer(strategy='median')",
            "numerical_scaler": "StandardScaler()",
            "categorical_imputer": "SimpleImputer(strategy='most_frequent')",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "leakage_isolation": "Fitted strictly on training partition (X_train) inside sklearn Pipeline"
        },
        "test_set_status": {
            "locked": True,
            "accessed": False,
            "statement": (
                "The test set was not used for baseline model fitting, preprocessing, "
                "model selection, threshold selection, or performance evaluation."
            )
        },
        "targets": {}
    }

    for target_name in TARGET_NAMES:
        _, target_metrics = train_and_evaluate_target(
            target_name=target_name,
            df_raw=df_raw,
            X_all=X_all,
            num_cols=num_cols,
            cat_cols=cat_cols
        )
        all_metrics["targets"][target_name] = target_metrics

    # 7. Save baseline metrics JSON
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] Baseline metrics successfully saved to: {OUTPUT_METRICS_PATH}")

    if args.all:
        run_baseline_v2()

    print("\n" + "=" * 70)
    print("Step 9: Baseline model training and validation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
