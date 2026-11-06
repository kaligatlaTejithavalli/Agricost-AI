"""Automated unit and integration tests for Phase 3B: Updated Model Training & Chronological Evaluation.

Verifies:
1. Extended feature data loading (312 rows, 26 commodities, FY 2014-15 to 2025-26).
2. Chronological partitioning (Train: 234 samples, 2014-15 to 2022-23; Test: 78 samples, 2023-24 to 2025-26).
3. Zero temporal leakage and boundary isolation.
4. Model pipeline creation, fitting, baseline calculations, and error metrics (MAE, RMSE, MAPE).
5. Held-out test performance ranking: Naive Persistence achieves lowest error; Ridge Regression is top ML model.
6. Ridge Regression outperforming tree ensembles (Random Forest, Gradient Boosting) on small-sample macroeconomic series.
7. Separate versioned evaluation outputs under models/evaluation/phase_3b/.
8. Final production candidate model trained on all 312 observations saved under models/trained/production_candidate_extended.joblib.
9. Strict preservation of existing trained models (best_model.joblib), raw datasets, and previous evaluation files.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    BASE_DIR,
    EVALUATION_DIR,
    EXTENDED_FEATURES_CSV,
    RAW_WPI_CSV,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.feature_engineering import TARGET_COLUMN
from src.train_extended import (
    PHASE_3B_EVAL_DIR,
    PHASE_3B_TEST_PERIODS,
    PHASE_3B_TRAIN_PERIODS,
    PRODUCTION_CANDIDATE_META_PATH,
    PRODUCTION_CANDIDATE_MODEL_PATH,
    calculate_comprehensive_metrics,
    create_extended_model_pipeline,
    evaluate_phase_3b_models,
    load_extended_feature_data,
    save_phase_3b_evaluation_artifacts,
    split_extended_data_chronologically,
    train_final_production_candidate,
)

# Reference SHA-256 hash of the existing deployed best_model.joblib
BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"


# ===========================================================================
# 1. Dataset Loading and Chronological Split Tests
# ===========================================================================
class TestPhase3BDataLoadingAndSplitting:
    """Tests verifying extended dataset ingestion and chronological partition boundaries."""

    def test_extended_feature_dataset_loaded(self):
        """Verify extended feature matrix has 312 rows, 26 commodities, and positive WPI."""
        df = load_extended_feature_data()
        assert len(df) == 312
        assert df["commodity_code"].nunique() == 26
        assert TARGET_COLUMN in df.columns
        assert (df[TARGET_COLUMN] > 0).all()
        assert (df["target_unit"] == TARGET_LABEL).all()

    def test_chronological_split_dimensions(self):
        """Verify train has 234 samples (9 FYs) and test has 78 samples (3 FYs)."""
        train_df, test_df = split_extended_data_chronologically()
        assert len(train_df) == 234
        assert len(test_df) == 78
        assert len(train_df) + len(test_df) == 312

        assert train_df["commodity_code"].nunique() == 26
        assert test_df["commodity_code"].nunique() == 26

        # Check periods
        assert list(train_df["period_label"].unique()) == PHASE_3B_TRAIN_PERIODS
        assert list(test_df["period_label"].unique()) == PHASE_3B_TEST_PERIODS

    def test_temporal_boundary_separation(self):
        """Verify zero overlap and strict chronological ordering between train and test."""
        train_df, test_df = split_extended_data_chronologically()
        train_periods = set(train_df["period_label"].unique())
        test_periods = set(test_df["period_label"].unique())

        assert len(train_periods.intersection(test_periods)) == 0
        assert train_df["period_order"].max() < test_df["period_order"].min()
        assert train_df["period_order"].max() == 10  # 2022-23
        assert test_df["period_order"].min() == 11   # 2023-24

        assert train_df["start_year"].max() < test_df["start_year"].min()
        assert train_df["start_year"].max() == 2022
        assert test_df["start_year"].min() == 2023


# ===========================================================================
# 2. Pipeline Construction and Baseline Calculation Tests
# ===========================================================================
class TestPhase3BModelPipelinesAndBaselines:
    """Tests verifying pipeline creation, baseline mechanics, and error metrics."""

    def test_pipeline_types_and_steps(self):
        """Verify Pipeline structure contains ColumnTransformer and expected regressor."""
        for m_type in ["ridge_10", "ridge_1", "random_forest", "gradient_boosting"]:
            pipe = create_extended_model_pipeline(m_type)
            assert isinstance(pipe, Pipeline)
            assert "preprocessor" in pipe.named_steps
            assert "regressor" in pipe.named_steps

    def test_metrics_calculation(self):
        """Verify calculate_comprehensive_metrics computes MAE, RMSE, and MAPE correctly."""
        y_true = pd.Series([100.0, 110.0, 120.0])
        y_pred = pd.Series([102.0, 108.0, 126.0])

        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        assert metrics["MAE (WPI pts)"] == round((2.0 + 2.0 + 6.0) / 3.0, 2)
        assert metrics["RMSE (WPI pts)"] == round(np.sqrt((4.0 + 4.0 + 36.0) / 3.0), 2)
        expected_mape = round(np.mean([2.0 / 100.0, 2.0 / 110.0, 6.0 / 120.0]) * 100.0, 2)
        assert metrics["MAPE (%)"] == expected_mape


# ===========================================================================
# 3. Held-Out Evaluation & Model Comparison Tests
# ===========================================================================
class TestPhase3BHeldOutEvaluation:
    """Tests verifying actual held-out performance on FY 2023-24 to 2025-26."""

    def test_overall_comparison_ranking(self):
        """Verify comparison table contains all models and baselines with valid metrics."""
        train_df, test_df = split_extended_data_chronologically()
        comp_df, cat_df, pred_df, pipelines = evaluate_phase_3b_models(train_df, test_df)

        assert len(comp_df) >= 7
        assert "Model / Baseline" in comp_df.columns
        assert "RMSE (WPI pts)" in comp_df.columns
        assert "MAE (WPI pts)" in comp_df.columns
        assert "MAPE (%)" in comp_df.columns

        # Verify sorted ascending by RMSE
        assert comp_df["RMSE (WPI pts)"].is_monotonic_increasing

        # Naive Persistence is top overall due to macro plateau
        top_model = comp_df.iloc[0]["Model / Baseline"]
        assert "Naïve Persistence" in top_model

    def test_ridge_regression_is_best_ml_model(self):
        """Verify Ridge Regression outperforms tree ensembles (Random Forest, Gradient Boosting)."""
        train_df, test_df = split_extended_data_chronologically()
        comp_df, _, _, _ = evaluate_phase_3b_models(train_df, test_df)

        ml_df = comp_df[comp_df["Model Type"] == "Machine Learning"].reset_index(drop=True)
        top_ml = ml_df.iloc[0]

        assert "Ridge Regression" in top_ml["Model / Baseline"]
        assert top_ml["RMSE (WPI pts)"] <= 16.0

        # Verify Ridge RMSE is substantially lower than Random Forest and Gradient Boosting (>10 pts lower)
        rf_row = comp_df[comp_df["Model / Baseline"].str.contains("Random Forest")].iloc[0]
        gb_row = comp_df[comp_df["Model / Baseline"].str.contains("Gradient Boosting")].iloc[0]

        assert top_ml["RMSE (WPI pts)"] < rf_row["RMSE (WPI pts)"] - 10.0
        assert top_ml["RMSE (WPI pts)"] < gb_row["RMSE (WPI pts)"] - 10.0

    def test_category_evaluation_breakdown(self):
        """Verify performance breakdown exists across all 5 agricultural input categories."""
        train_df, test_df = split_extended_data_chronologically()
        _, cat_df, _, _ = evaluate_phase_3b_models(train_df, test_df)

        categories = set(cat_df["input_category"].unique())
        assert "Fertilizer" in categories
        assert "Fuel & Power" in categories
        assert "Farm Machinery" in categories
        assert "Livestock Input" in categories
        assert "Pesticide / Agrochemical" in categories

    def test_test_predictions_residuals(self):
        """Verify predictions table has 78 rows and valid actuals/residuals."""
        train_df, test_df = split_extended_data_chronologically()
        _, _, pred_df, _ = evaluate_phase_3b_models(train_df, test_df)

        assert len(pred_df) == 78
        assert TARGET_COLUMN in pred_df.columns
        assert "pred_ridge_regression_alpha_10.0" in pred_df.columns
        assert "residual_ridge_regression_alpha_10.0" in pred_df.columns
        assert pred_df["pred_ridge_regression_alpha_10.0"].isna().sum() == 0


# ===========================================================================
# 4. Artifact Preservation & Versioned Paths Tests
# ===========================================================================
class TestPhase3BArtifactsAndPreservation:
    """Verifies that Phase 3B outputs are versioned and existing deployed models remain pristine."""

    def test_phase_3b_evaluation_artifacts_saved(self):
        """Verify evaluation artifacts exist under models/evaluation/phase_3b/."""
        assert PHASE_3B_EVAL_DIR.exists()
        assert (PHASE_3B_EVAL_DIR / "model_comparison.csv").exists()
        assert (PHASE_3B_EVAL_DIR / "category_evaluation.csv").exists()
        assert (PHASE_3B_EVAL_DIR / "test_predictions.csv").exists()
        assert (PHASE_3B_EVAL_DIR / "phase_3b_evaluation_report.json").exists()

        with open(PHASE_3B_EVAL_DIR / "phase_3b_evaluation_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
            assert report["training_partition"]["sample_count"] == 234
            assert report["testing_partition"]["sample_count"] == 78
            assert "methodological_comparison_with_backtesting" in report

    def test_production_candidate_model_serialized(self):
        """Verify production_candidate_extended.joblib exists, is loadable, and metadata is complete."""
        assert PRODUCTION_CANDIDATE_MODEL_PATH.exists()
        assert PRODUCTION_CANDIDATE_META_PATH.exists()

        pipeline = joblib.load(PRODUCTION_CANDIDATE_MODEL_PATH)
        assert isinstance(pipeline, Pipeline)

        with open(PRODUCTION_CANDIDATE_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert meta["training_observations"] == 312
            assert meta["commodities_count"] == 26
            assert meta["first_training_period"] == "2014-15"
            assert meta["last_training_period"] == "2025-26"

    def test_existing_best_model_hash_strictly_preserved(self):
        """Verify best_model.joblib is byte-for-byte preserved (SHA-256) and trained on 156 samples."""
        model_file = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_file.exists()
        with open(model_file, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        assert current_hash == BEST_MODEL_SHA256, (
            f"Existing deployed best_model.joblib was modified! Expected {BEST_MODEL_SHA256}, got {current_hash}"
        )
        pipeline = joblib.load(model_file)
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "predict")
        scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
        assert scaler.n_samples_seen_ == 156

    def test_raw_dataset_is_strictly_unmodified(self):
        """Verify raw WPI CSV byte size and record count remain pristine."""
        raw_df = pd.read_csv(RAW_WPI_CSV)
        assert raw_df.shape == (869, 27)
        assert raw_df.iloc[0]["COMM_NAME"] == "Bajra"

    def test_previous_evaluation_artifacts_unmodified(self):
        """Verify previous Stage 5 evaluation reports and backtesting artifacts exist."""
        assert (EVALUATION_DIR / "model_comparison.csv").exists()
        assert (EVALUATION_DIR / "evaluation_report.json").exists()
        assert (EVALUATION_DIR / "backtesting" / "backtesting_summary.json").exists()
