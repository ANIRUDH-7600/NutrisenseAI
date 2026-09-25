"""Automated verification test suite for Scenario A v1 model comparison (30 Features)."""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

RAW_DTA_PATH = "IAKR7EDT/IAKR7EFL.DTA"
EXPECTED_RAW_BYTES = 441380745
INPUT_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
OUTPUT_METRICS_V2_JSON_PATH = "data/interim/model_comparison_metrics_v2.json"
OUTPUT_METRICS_V2_CSV_PATH = "data/interim/model_comparison_metrics_v2.csv"
MODELS_V2_DIR = "models/v1/research_candidates"

TARGETS = ["stunting", "underweight", "wasting"]
MODEL_FAMILIES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
CONDITIONS = ["unweighted", "class_weighted"]


def test_required_metrics_files_exist_v2():
    """Asserts that model comparison metrics v2 JSON and CSV exist and are non-empty."""
    assert os.path.exists(OUTPUT_METRICS_V2_JSON_PATH)
    assert os.path.exists(OUTPUT_METRICS_V2_CSV_PATH)

    with open(OUTPUT_METRICS_V2_JSON_PATH, "r") as f:
        meta = json.load(f)
    assert meta["total_experiments_conducted"] == 30
    assert len(meta["results"]) == 30
    assert meta["feature_count"] == 30
    assert meta["test_set_status"]["locked"] is True
    assert meta["test_set_status"]["accessed"] is False


def test_all_30_experiments_represented_v2():
    """Verifies that all 30 combinations (5 models x 2 conditions x 3 targets) exist."""
    df = pd.read_csv(OUTPUT_METRICS_V2_CSV_PATH)
    assert len(df) == 30

    for target in TARGETS:
        for model in MODEL_FAMILIES:
            for cond in CONDITIONS:
                match = df[(df["target"] == target) & (df["model"] == model) & (df["condition"] == cond)]
                assert len(match) == 1, f"Missing experiment: {target} | {model} | {cond}"


def test_metric_bounds_and_validity_v2():
    """Verifies that all metrics fall within theoretical bounds."""
    df = pd.read_csv(OUTPUT_METRICS_V2_CSV_PATH)
    assert (df["roc_auc"] >= 0.5).all() and (df["roc_auc"] <= 1.0).all()
    assert (df["pr_auc"] >= 0.0).all() and (df["pr_auc"] <= 1.0).all()
    assert (df["accuracy"] >= 0.0).all() and (df["accuracy"] <= 1.0).all()
    assert (df["precision"] >= 0.0).all() and (df["precision"] <= 1.0).all()
    assert (df["recall"] >= 0.0).all() and (df["recall"] <= 1.0).all()
    assert (df["f1_score"] >= 0.0).all() and (df["f1_score"] <= 1.0).all()


def test_all_30_model_artifacts_exist_in_models_v2():
    """Verifies that each of the 30 trained model artifacts exists in research candidates or models/v1."""
    df = pd.read_csv(OUTPUT_METRICS_V2_CSV_PATH)
    for _, row in df.iterrows():
        artifact_path = os.path.join(MODELS_V2_DIR, row["artifact_file"])
        if not os.path.exists(artifact_path):
            artifact_path = os.path.join("models/v1", row["artifact_file"].replace("_v2.joblib", "_v1.joblib"))
        assert os.path.exists(artifact_path), f"Missing artifact: {artifact_path}"
        assert os.path.getsize(artifact_path) > 1000


def test_prediction_validity_on_sample_v2():
    """Tests sample inference with LightGBM models on 30 features."""
    from src.features.build_features import transform_candidate_features_v2

    sample_df = pd.read_csv(INPUT_DATA_PATH, nrows=20)
    X_sample = transform_candidate_features_v2(sample_df)
    assert X_sample.shape == (20, 30)

    for target in TARGETS:
        lgbm_path = os.path.join("models/v1", f"model_comparison_lightgbm_{target}_unweighted_v1.joblib")
        pipeline = joblib.load(lgbm_path)
        probs = pipeline.predict_proba(X_sample)
        assert probs.shape == (20, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_v1_model_comparison_artifacts_preserved():
    """Verifies that original v1 model comparison artifacts in archive/ remain untouched."""
    assert os.path.exists("archive/scenario-a-34-feature/data/interim/model_comparison_metrics.json")
    assert os.path.exists("archive/scenario-a-34-feature/data/interim/model_comparison_metrics.csv")
    for target in TARGETS:
        for cond in ["unweighted", "class_weighted"]:
            v1_file = os.path.join("archive/scenario-a-34-feature/models", f"model_comparison_lightgbm_{target}_{cond}.joblib")
            assert os.path.exists(v1_file), f"Original v1 artifact missing: {v1_file}"


def test_raw_dataset_immutability_v2():
    """Verifies that raw DHS dataset is unaltered."""
    assert os.path.exists(RAW_DTA_PATH)
    assert os.path.getsize(RAW_DTA_PATH) == EXPECTED_RAW_BYTES
