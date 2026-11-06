"""Unit tests for Stage 5 Model Training and Evaluation module (src/train.py).

Verifies:
1. Training and testing dataset ingestion and chronological partition integrity.
2. Naive Persistence baseline and Drift baseline mathematical correctness.
3. MAE and RMSE error metric calculations in WPI index points.
4. Model pipeline creation, fitting, and prediction across Ridge, Random Forest, and Gradient Boosting.
5. Prevention of target leakage and consistent train/test preprocessing.
6. Model comparison ranking against Naive Persistence.
7. Persistence of trained estimators loadable via joblib without retraining.
8. Preservation of 'WPI index points' units in all model metadata and evaluation artifacts.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    EVALUATION_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.feature_engineering import CORE_FEATURE_COLUMNS, TARGET_COLUMN
from src.train import (
    calculate_metrics,
    create_model_pipeline,
    evaluate_all_models,
    load_training_and_testing_data,
    predict_drift_baseline,
    predict_naive_persistence,
    predict_rolling_mean_baseline,
    run_training_pipeline,
)


class TestTrainingDataIngestion:
    """Tests for load_training_and_testing_data() verifying dataset health and boundaries."""

    def test_record_counts_and_commodities(self):
        """Verify train (156 samples) and test (78 samples) dimensions and commodity coverage."""
        train_df, test_df = load_training_and_testing_data()

        assert len(train_df) == 156
        assert len(test_df) == 78
        assert train_df["commodity_code"].nunique() == 26
        assert test_df["commodity_code"].nunique() == 26

    def test_target_variable_integrity(self):
        """Verify target variable is strictly positive and labeled as 'WPI index points'."""
        train_df, test_df = load_training_and_testing_data()

        for partition in [train_df, test_df]:
            assert TARGET_COLUMN in partition.columns
            assert (partition[TARGET_COLUMN] > 0).all()
            assert (partition["target_unit"] == TARGET_LABEL).all()
            assert (partition["target_unit"] == "WPI index points").all()

    def test_chronological_periods_disjoint(self):
        """Verify training periods (2014-15 to 2019-20) and testing periods (2020-21 to 2022-23) do not overlap."""
        train_df, test_df = load_training_and_testing_data()

        train_periods = set(train_df["period_label"].unique())
        test_periods = set(test_df["period_label"].unique())

        assert len(train_periods.intersection(test_periods)) == 0
        assert train_df["period_order"].max() < test_df["period_order"].min()
        assert train_df["period_order"].max() == 7
        assert test_df["period_order"].min() == 8


class TestBaselineForecastingModels:
    """Tests for Naive Persistence, Drift, and Rolling Mean baseline forecasts."""

    def test_naive_persistence_logic(self):
        """Verify Naive Persistence predicts exactly y_t = lag_1."""
        _, test_df = load_training_and_testing_data()
        preds = predict_naive_persistence(test_df)

        assert (preds == test_df["lag_1"]).all()
        # Verify known held-out test error metrics
        metrics = calculate_metrics(test_df[TARGET_COLUMN], preds)
        assert metrics["MAE"] == 14.01
        assert metrics["RMSE"] == 22.02

    def test_drift_baseline_logic(self):
        """Verify Drift Baseline formula: lag_1 + (lag_1 - lag_2)."""
        _, test_df = load_training_and_testing_data()
        preds = predict_drift_baseline(test_df)

        expected = (test_df["lag_1"] + (test_df["lag_1"] - test_df["lag_2"])).round(2)
        assert np.allclose(preds, expected)

        metrics = calculate_metrics(test_df[TARGET_COLUMN], preds)
        assert metrics["MAE"] == 11.92
        assert metrics["RMSE"] == 20.49

    def test_rolling_mean_baseline_logic(self):
        """Verify Rolling Mean Baseline equals rolling_mean_2."""
        _, test_df = load_training_and_testing_data()
        preds = predict_rolling_mean_baseline(test_df)

        assert (preds == test_df["rolling_mean_2"]).all()
        metrics = calculate_metrics(test_df[TARGET_COLUMN], preds)
        assert metrics["MAE"] == 16.50
        assert metrics["RMSE"] == 25.83


class TestMetricsCalculation:
    """Tests for calculate_metrics() with known synthetic and actual vectors."""

    def test_metrics_mathematical_accuracy(self):
        """Verify MAE and RMSE on known vectors."""
        y_true = pd.Series([100.0, 110.0, 120.0])
        y_pred = pd.Series([102.0, 106.0, 125.0])
        # Errors: |2|, |-4|, |5| -> MAE = (2 + 4 + 5) / 3 = 11/3 = 3.67
        # Squared errors: 4, 16, 25 -> Mean: 45 / 3 = 15 -> RMSE = sqrt(15) = 3.87
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["MAE"] == 3.67
        assert metrics["RMSE"] == 3.87


class TestModelTrainingAndPipelines:
    """Tests for scikit-learn model pipelines and evaluation logic."""

    def test_pipeline_construction_and_fitting(self):
        """Verify constructing and fitting Ridge, Random Forest, and Gradient Boosting."""
        train_df, test_df = load_training_and_testing_data()
        y_train = train_df[TARGET_COLUMN]

        for m_type in ["ridge", "random_forest", "gradient_boosting"]:
            pipe = create_model_pipeline(m_type, random_state=42)
            pipe.fit(train_df, y_train)

            preds = pipe.predict(test_df)
            assert len(preds) == len(test_df)
            assert not np.isnan(preds).any()
            assert (preds > 0).all()

    def test_reject_unsupported_model_type(self):
        """Reject invalid model type strings."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            create_model_pipeline("deep_neural_network")

    def test_no_target_leakage_in_features(self):
        """Verify target column wpi_index_points is never included in the input feature matrix."""
        train_df, _ = load_training_and_testing_data()
        pipe = create_model_pipeline("ridge")

        # Check transformed feature names from ColumnTransformer
        preprocessor = pipe.named_steps["preprocessor"]
        preprocessor.fit(train_df)

        feature_names = preprocessor.get_feature_names_out()
        for name in feature_names:
            assert TARGET_COLUMN not in name, f"Target column leaked into feature: {name}"

    def test_evaluate_all_models_output(self):
        """Verify overall model comparison output and rankings."""
        train_df, test_df = load_training_and_testing_data()
        comparison_df, category_df, pred_df, pipelines = evaluate_all_models(train_df, test_df)

        # Must evaluate all 6 models / baselines
        assert len(comparison_df) == 6
        expected_models = {
            "Naive Persistence (y_t = lag_1)",
            "Drift Baseline (Linear Momentum)",
            "Rolling Mean Baseline (2-Year MA)",
            "Ridge Regression (alpha=10.0)",
            "Random Forest (depth=3)",
            "Gradient Boosting (depth=2)",
        }
        assert set(comparison_df["Model / Baseline"]) == expected_models

        # Check Ridge beats Naive Persistence
        ridge_row = comparison_df[comparison_df["Model / Baseline"] == "Ridge Regression (alpha=10.0)"].iloc[0]
        assert ridge_row["RMSE (WPI points)"] == 21.03
        assert ridge_row["MAE (WPI points)"] == 12.76
        assert ridge_row["Beats Naive Persistence"] is True or ridge_row["Beats Naive Persistence"] == True

        # Check Drift baseline has lowest RMSE
        top_model = comparison_df.iloc[0]
        assert top_model["Model / Baseline"] == "Drift Baseline (Linear Momentum)"
        assert top_model["RMSE (WPI points)"] == 20.49

        # Predictions table check
        assert len(pred_df) == 78
        assert "pred_ridge_regression_alpha_10.0" in pred_df.columns
        assert "residual_ridge_regression_alpha_10.0" in pred_df.columns


class TestModelArtifactPersistenceAndReload:
    """Tests verifying saved artifacts can be loaded and executed without retraining."""

    def test_saved_model_loadable_and_reproducible(self):
        """Verify best_model.joblib is loadable via joblib.load and reproduces test predictions."""
        model_path = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_path.exists()
        assert model_path.stat().st_size > 0

        # Load saved pipeline
        loaded_model = joblib.load(model_path)
        _, test_df = load_training_and_testing_data()

        # Run inference
        loaded_preds = loaded_model.predict(test_df)
        assert len(loaded_preds) == 78

        # Compare with predictions saved in CSV
        pred_csv = EVALUATION_DIR / "test_predictions.csv"
        saved_preds_df = pd.read_csv(pred_csv)
        assert np.allclose(loaded_preds.round(2), saved_preds_df["pred_ridge_regression_alpha_10.0"])

    def test_model_metadata_and_units_integrity(self):
        """Verify model_metadata.json documents WPI index points, Drift best overall, and Ridge ML pipeline."""
        meta_path = TRAINED_MODELS_DIR / "model_metadata.json"
        assert meta_path.exists()

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["target_variable"] == "wpi_index_points"
        assert metadata["target_unit"] == "WPI index points"
        assert metadata["training_samples"] == 156
        assert metadata["testing_samples"] == 78
        assert metadata["commodities_count"] == 26
        assert "One-step-ahead" in metadata["evaluation_type"]

        # Verify explicit distinction between Best Overall Method and Selected ML Pipeline
        assert metadata["best_overall_method"] == "Drift Baseline (Linear Momentum)"
        assert metadata["best_overall_rmse"] == 20.49
        assert metadata["best_overall_mae"] == 11.92
        assert metadata["selected_ml_pipeline_name"] == "Ridge Regression (alpha=10.0)"
        assert metadata["selected_ml_pipeline_artifact"] == "models/trained/best_model.joblib"
        assert metadata["selected_ml_rmse"] == 21.03
        assert metadata["selected_ml_mae"] == 12.76
        assert "Drift Baseline" in metadata["model_selection_clarification"]

    def test_drift_baseline_results_preserved_in_evaluation(self):
        """Verify Drift Baseline results are fully preserved in evaluation report and predictions."""
        report_path = EVALUATION_DIR / "evaluation_report.json"
        assert report_path.exists()

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["best_overall_method"]["name"] == "Drift Baseline (Linear Momentum)"
        assert report["best_overall_method"]["rmse"] == 20.49
        assert report["selected_ml_pipeline"]["name"] == "Ridge Regression (alpha=10.0)"
        assert report["selected_ml_pipeline"]["rmse"] == 21.03
        assert report["drift_baseline_preserved"] is True

        # Verify Drift predictions are in test_predictions.csv
        preds_df = pd.read_csv(EVALUATION_DIR / "test_predictions.csv")
        assert "pred_drift_baseline_linear_momentum" in preds_df.columns
        assert not preds_df["pred_drift_baseline_linear_momentum"].isna().any()
