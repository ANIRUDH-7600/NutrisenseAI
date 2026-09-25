"""Automated verification test suite for Scenario A v2: Baseline Model (30 Features)."""
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
FEATURE_METADATA_V2_PATH = "data/interim/feature_metadata_v2.json"
SPLIT_METADATA_PATH = "data/interim/split_metadata.json"
BASELINE_METRICS_V2_PATH = "data/interim/baseline_metrics_v2.json"
MODELS_V2_DIR = "models/v1/research_candidates"
TARGETS = ["stunting", "underweight", "wasting"]


def test_required_files_and_artifacts_exist_v2():
    """Asserts required data files, metrics, and trained v2 model artifacts exist."""
    assert os.path.exists(INPUT_DATA_PATH), f"Missing {INPUT_DATA_PATH}"
    assert os.path.exists(FEATURE_METADATA_V2_PATH), f"Missing {FEATURE_METADATA_V2_PATH}"
    assert os.path.exists(SPLIT_METADATA_PATH), f"Missing {SPLIT_METADATA_PATH}"
    assert os.path.exists(BASELINE_METRICS_V2_PATH), f"Missing {BASELINE_METRICS_V2_PATH}"

    for target in TARGETS:
        model_file = os.path.join(MODELS_V2_DIR, f"baseline_logistic_{target}_v2.joblib")
        assert os.path.exists(model_file), f"Missing model artifact: {model_file}"
        assert os.path.getsize(model_file) > 1000, f"Model artifact {model_file} is suspiciously small!"


def test_raw_dataset_immutability_v2():
    """Asserts raw .DTA file has not been altered or overwritten."""
    assert os.path.exists(RAW_DTA_PATH), f"Raw dataset {RAW_DTA_PATH} missing!"
    current_size = os.path.getsize(RAW_DTA_PATH)
    assert current_size == EXPECTED_RAW_BYTES, (
        f"Raw .DTA altered! Expected {EXPECTED_RAW_BYTES} bytes, found {current_size} bytes."
    )


def test_feature_registry_adherence_and_isolation_v2():
    """Asserts that exactly 30 candidate features are used in v2 with zero leakage or sensitive features."""
    with open(FEATURE_METADATA_V2_PATH, "r") as f:
        f_meta = json.load(f)
    approved_feats = set(f_meta["features"])
    assert len(approved_feats) == 30, f"Expected 30 features, found {len(approved_feats)}"

    prohibited = {
        "v001", "v002", "v003", "v005", "hh_id", "mother_id", "split",
        "stunting", "underweight", "wasting", "stunting_severe", "underweight_severe", "wasting_severe",
        "eligible_stunting", "eligible_underweight", "eligible_wasting", "eligible_complete_unified",
        "hw70", "hw71", "hw72", "hw73", "hw70_clean", "hw71_clean", "hw72_clean", "hw73_clean",
        "hw2", "hw3", "hw2_clean", "hw3_clean", "hw4", "hw5", "hw6", "hw7", "hw8", "hw9", "hw10", "hw11", "hw12",
        "household_head_female", "caste_category", "religion_category", "mother_education_level"
    }
    for var in prohibited:
        assert var not in approved_feats, f"PROHIBITED VARIABLE: '{var}' found in v2 feature set!"


def test_pipeline_encapsulation_and_preprocessing_v2():
    """Verifies that each saved v2 model artifact is an sklearn Pipeline with preprocessing included."""
    for target in TARGETS:
        model_path = os.path.join(MODELS_V2_DIR, f"baseline_logistic_{target}_v2.joblib")
        pipeline = joblib.load(model_path)
        assert hasattr(pipeline, "named_steps"), f"{model_path} is not a valid Pipeline!"
        assert "preprocessor" in pipeline.named_steps, f"Preprocessor step missing from {model_path}"
        assert "classifier" in pipeline.named_steps, f"Classifier step missing from {model_path}"
        
        prep = pipeline.named_steps["preprocessor"]
        transformer_names = [t[0] for t in prep.transformers]
        assert "num" in transformer_names, "Numerical transformer missing in ColumnTransformer"
        assert "cat" in transformer_names, "Categorical transformer missing in ColumnTransformer"


def test_baseline_metrics_structure_and_completeness_v2():
    """Asserts that baseline_metrics_v2.json contains complete, valid metrics for all 3 targets."""
    with open(BASELINE_METRICS_V2_PATH, "r") as f:
        metrics = json.load(f)

    assert metrics["model_family"] == "LogisticRegression"
    assert metrics["feature_count"] == 30
    assert metrics["test_set_status"]["locked"] is True
    assert metrics["test_set_status"]["accessed"] is False

    required_metric_keys = [
        "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "confusion_matrix"
    ]

    for target in TARGETS:
        assert target in metrics["targets"], f"Target {target} missing from baseline_metrics_v2.json"
        t_data = metrics["targets"][target]

        assert t_data["train_sample_count"] > 140000
        assert t_data["validation_sample_count"] > 30000
        assert t_data["features_before_preprocessing"] == 30
        assert t_data["features_after_preprocessing"] == 77

        t_metrics = t_data["metrics"]
        for m_key in required_metric_keys:
            assert m_key in t_metrics, f"Metric {m_key} missing for target {target}"

        assert 0.0 <= t_metrics["accuracy"] <= 1.0
        assert 0.0 <= t_metrics["precision"] <= 1.0
        assert 0.0 <= t_metrics["recall"] <= 1.0
        assert 0.0 <= t_metrics["f1_score"] <= 1.0
        assert 0.5 <= t_metrics["roc_auc"] <= 1.0
        assert 0.0 <= t_metrics["pr_auc"] <= 1.0

        cm = t_metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2
        total_cm = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total_cm == t_data["validation_sample_count"]


def test_test_set_lock_principle_adherence_v2():
    """Asserts that test partition records were never touched in baseline_metrics_v2.json."""
    with open(BASELINE_METRICS_V2_PATH, "r") as f:
        metrics = json.load(f)

    for target in TARGETS:
        t_data = metrics["targets"][target]
        assert "test_sample_count" not in t_data
        assert "test_metrics" not in t_data


def test_model_prediction_validity_on_sample_v2():
    """Verifies that loaded v2 pipelines produce valid probabilities [0, 1] on 30-feature inputs."""
    from src.features.build_features import transform_candidate_features_v2

    df_sample = pd.read_csv(INPUT_DATA_PATH, nrows=50)
    X_sample = transform_candidate_features_v2(df_sample)
    assert X_sample.shape == (50, 30)

    for target in TARGETS:
        model_path = os.path.join(MODELS_V2_DIR, f"baseline_logistic_{target}_v2.joblib")
        pipeline = joblib.load(model_path)

        probs = pipeline.predict_proba(X_sample)
        preds = pipeline.predict(X_sample)

        assert probs.shape == (50, 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert set(np.unique(preds)).issubset({0, 1})


def test_v1_baseline_artifacts_preserved():
    """Verifies that original v1 baseline artifacts remain untouched in archive/."""
    for target in TARGETS:
        model_file = os.path.join("archive/scenario-a-34-feature/models", f"baseline_logistic_{target}.joblib")
        assert os.path.exists(model_file), f"Original v1 baseline artifact missing: {model_file}"
        assert os.path.getsize(model_file) > 1000
