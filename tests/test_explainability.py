"""Unit tests for Stage 6 Explainable AI (XAI) using authentic SHAP (src/explainability.py).

Test Suite Verifications:
1. Loading the saved Ridge Regression model from models/trained/best_model.joblib without retraining.
2. Loading and validating the fitted preprocessing pipeline (ColumnTransformer).
3. Exact matching between transformed feature names and SHAP explanation dimensions.
4. Correct SHAP explanation output dimensions (78 test rows x 12 features).
5. All SHAP attributions and expected base values are strictly finite real numbers (no NaNs or Infs).
6. Exact mathematical SHAP additivity across all held-out test predictions (Base + Sum of SHAP == y_hat).
7. Individual prediction explanations (specifically Urea FY 2022-23) with correct metadata.
8. Preservation and enforcement of 'WPI index points' units across all outputs (anti-rupee guardrail).
9. Robust error handling for invalid commodity codes or unavailable financial periods.
10. Verification that model weights and serialized file hash remain completely unchanged (no retraining).
11. Global feature importance calculation and ranking consistency.
12. Serialization and health check of all saved SHAP artifacts and plots in models/evaluation/shap/.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    SHAP_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.explainability import (
    SHAP_CORRELATED_FEATURES_DISCLAIMER,
    SHAP_UNIT_LABEL,
    SHAP_WPI_VS_RUPEES_DISCLAIMER,
    clean_feature_name_for_display,
    compute_global_feature_importance,
    compute_shap_explanations,
    create_linear_explainer,
    explain_individual_prediction,
    extract_pipeline_components,
    get_transformed_feature_names,
    load_trained_pipeline,
    prepare_preprocessed_matrices,
    run_explainability_pipeline,
)
from src.feature_engineering import TARGET_COLUMN
from src.train import load_training_and_testing_data

# Expected SHA-256 hash of the fitted model artifact to guarantee zero retraining
BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Fixture providing the loaded trained pipeline."""
    return load_trained_pipeline()


@pytest.fixture(scope="module")
def train_test_datasets():
    """Fixture providing train and test DataFrames."""
    return load_training_and_testing_data()


@pytest.fixture(scope="module")
def preprocessed_shap_artifacts(loaded_pipeline, train_test_datasets):
    """Fixture providing computed SHAP explainer, explanations, and feature names."""
    train_df, test_df = train_test_datasets
    preprocessor, regressor = extract_pipeline_components(loaded_pipeline)
    X_train_trans, X_test_trans, feature_names = prepare_preprocessed_matrices(
        preprocessor=preprocessor,
        train_df=train_df,
        test_df=test_df,
    )
    explainer = create_linear_explainer(regressor, X_train_trans, feature_names)
    shap_explanation = compute_shap_explanations(explainer, X_test_trans)
    return {
        "pipeline": loaded_pipeline,
        "preprocessor": preprocessor,
        "regressor": regressor,
        "train_df": train_df,
        "test_df": test_df,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "feature_names": feature_names,
        "explainer": explainer,
        "shap_explanation": shap_explanation,
    }


class TestModelAndPipelineLoading:
    """Tests 1 & 2: Loading saved model and preprocessing pipeline without retraining."""

    def test_load_trained_pipeline_type_and_steps(self, loaded_pipeline):
        """Verify pipeline is a scikit-learn Pipeline with preprocessor and regressor."""
        assert isinstance(loaded_pipeline, Pipeline)
        assert "preprocessor" in loaded_pipeline.named_steps
        assert "regressor" in loaded_pipeline.named_steps

    def test_regressor_is_fitted_ridge(self, loaded_pipeline):
        """Verify the regressor is a fitted Ridge model with alpha=10.0."""
        regressor = loaded_pipeline.named_steps["regressor"]
        assert isinstance(regressor, Ridge)
        assert regressor.alpha == 10.0
        assert hasattr(regressor, "coef_")
        assert hasattr(regressor, "intercept_")
        assert regressor.coef_.ndim == 1
        assert len(regressor.coef_) == 12

    def test_preprocessor_is_fitted_column_transformer(self, loaded_pipeline):
        """Verify preprocessor is a fitted ColumnTransformer with correct feature output."""
        preprocessor = loaded_pipeline.named_steps["preprocessor"]
        assert isinstance(preprocessor, ColumnTransformer)
        feature_names = get_transformed_feature_names(preprocessor)
        assert len(feature_names) == 12
        assert "num__lag_1" in feature_names
        assert "num__lag_2" in feature_names
        assert "num__rolling_mean_2" in feature_names
        assert "num__historical_pct_change" in feature_names
        assert "num__period_order" in feature_names


class TestShapDimensionsAndFeatures:
    """Tests 3, 4 & 5: Matching feature names, output dimensions, and finite numerical values."""

    def test_feature_names_match_shap_dimension(self, preprocessed_shap_artifacts):
        """Verify 1-to-1 match between transformed feature names and SHAP column count."""
        artifacts = preprocessed_shap_artifacts
        feature_names = artifacts["feature_names"]
        shap_explanation = artifacts["shap_explanation"]

        assert len(feature_names) == 12
        assert shap_explanation.values.shape[1] == 12
        assert list(shap_explanation.feature_names) == feature_names

    def test_shap_output_dimensions(self, preprocessed_shap_artifacts):
        """Verify SHAP explanation covers all 78 held-out test observations across 12 features."""
        shap_explanation = preprocessed_shap_artifacts["shap_explanation"]
        assert shap_explanation.values.shape == (78, 12)
        assert len(shap_explanation.base_values) == 78

    def test_all_shap_values_finite(self, preprocessed_shap_artifacts):
        """Verify no NaN, Inf, or null values exist in SHAP values or base values."""
        shap_explanation = preprocessed_shap_artifacts["shap_explanation"]
        assert np.all(np.isfinite(shap_explanation.values))
        assert np.all(np.isfinite(shap_explanation.base_values))


class TestShapAdditivity:
    """Test 6: Verify exact SHAP additivity across all test observations."""

    def test_exact_shap_additivity_all_test_observations(self, preprocessed_shap_artifacts):
        """Verify: Base Value + sum(phi_i) == y_hat within tight floating point tolerance."""
        artifacts = preprocessed_shap_artifacts
        pipeline = artifacts["pipeline"]
        test_df = artifacts["test_df"]
        shap_explanation = artifacts["shap_explanation"]

        preds = pipeline.predict(test_df)
        base_vals = shap_explanation.base_values
        shap_sums = np.sum(shap_explanation.values, axis=1)
        reconstructed = base_vals + shap_sums

        differences = np.abs(preds - reconstructed)
        max_diff = float(np.max(differences))

        # Regularized linear model should achieve near machine-precision additivity (< 1e-6)
        assert max_diff < 1e-6, f"SHAP additivity violated! Max discrepancy: {max_diff}"


class TestIndividualPredictionExplanation:
    """Test 7: Local explanation for specific historical prediction (Urea 2022-23)."""

    def test_local_explanation_urea_2022_23(self, preprocessed_shap_artifacts):
        """Verify local explanation for Urea in FY 2022-23."""
        artifacts = preprocessed_shap_artifacts
        urea_explanation = explain_individual_prediction(
            pipeline=artifacts["pipeline"],
            explainer=artifacts["explainer"],
            shap_explanation=artifacts["shap_explanation"],
            test_df=artifacts["test_df"],
            commodity_code=1310020001,
            period_label="2022-23",
        )

        assert urea_explanation["commodity_code"] == 1310020001
        assert urea_explanation["commodity_name"] == "Urea"
        assert urea_explanation["input_category"] == "Fertilizer"
        assert urea_explanation["period_label"] == "2022-23"
        assert urea_explanation["actual_wpi"] == pytest.approx(115.1, abs=0.1)
        assert urea_explanation["predicted_wpi"] == pytest.approx(115.76, abs=0.1)
        assert urea_explanation["additivity_verified"] is True
        assert urea_explanation["absolute_difference"] < 1e-4
        assert len(urea_explanation["feature_contributions"]) == 12

        # Check top negative contribution for Urea 2022-23 (lag_1 or rolling_mean_2 due to lower index)
        top_contrib = urea_explanation["feature_contributions"][0]
        assert "feature_raw" in top_contrib
        assert "feature_display" in top_contrib
        assert "shap_value" in top_contrib
        assert top_contrib["unit"] == "WPI index points"


class TestUnitLabelsAndDisclaimers:
    """Test 8: Strict WPI index point labeling and anti-rupee guardrail."""

    def test_unit_labels_in_explanations(self, preprocessed_shap_artifacts):
        """Verify all explainability outputs explicitly use 'WPI index points'."""
        artifacts = preprocessed_shap_artifacts
        global_df = compute_global_feature_importance(
            artifacts["shap_explanation"], artifacts["feature_names"]
        )

        assert (global_df["unit"] == TARGET_LABEL).all()
        assert (global_df["unit"] == "WPI index points").all()

        urea_explanation = explain_individual_prediction(
            pipeline=artifacts["pipeline"],
            explainer=artifacts["explainer"],
            shap_explanation=artifacts["shap_explanation"],
            test_df=artifacts["test_df"],
            commodity_code=1310020001,
            period_label="2022-23",
        )
        assert urea_explanation["unit"] == "WPI index points"
        for item in urea_explanation["feature_contributions"]:
            assert item["unit"] == "WPI index points"
            # Ensure no rupee or currency symbols are present in units
            assert "₹" not in item["unit"]
            assert "Rs" not in item["unit"]
            assert "rupee" not in item["unit"].lower()

    def test_disclaimers_present(self, preprocessed_shap_artifacts):
        """Verify correlation and unit distinction disclaimers are present."""
        artifacts = preprocessed_shap_artifacts
        urea_explanation = explain_individual_prediction(
            pipeline=artifacts["pipeline"],
            explainer=artifacts["explainer"],
            shap_explanation=artifacts["shap_explanation"],
            test_df=artifacts["test_df"],
            commodity_code=1310020001,
            period_label="2022-23",
        )
        assert "disclaimer_correlated_features" in urea_explanation
        assert "collinear" in urea_explanation["disclaimer_correlated_features"].lower()
        assert "causal" in urea_explanation["disclaimer_correlated_features"].lower()

        assert "disclaimer_units" in urea_explanation
        assert "WPI index points" in urea_explanation["disclaimer_units"]
        assert "rupee" in urea_explanation["disclaimer_units"].lower()


class TestErrorHandling:
    """Test 9: Handling invalid commodity selections and out-of-range periods."""

    def test_invalid_commodity_code_raises_error(self, preprocessed_shap_artifacts):
        """Verify ValueError is raised when querying a non-existent commodity code."""
        artifacts = preprocessed_shap_artifacts
        with pytest.raises(ValueError, match="Record not found"):
            explain_individual_prediction(
                pipeline=artifacts["pipeline"],
                explainer=artifacts["explainer"],
                shap_explanation=artifacts["shap_explanation"],
                test_df=artifacts["test_df"],
                commodity_code=9999999999,
                period_label="2022-23",
            )

    def test_invalid_period_raises_error(self, preprocessed_shap_artifacts):
        """Verify ValueError is raised when querying an unavailable financial year."""
        artifacts = preprocessed_shap_artifacts
        with pytest.raises(ValueError, match="Record not found"):
            explain_individual_prediction(
                pipeline=artifacts["pipeline"],
                explainer=artifacts["explainer"],
                shap_explanation=artifacts["shap_explanation"],
                test_df=artifacts["test_df"],
                commodity_code=1310020001,
                period_label="2099-00",
            )


class TestModelIntegrityAndUnchangedWeights:
    """Test 10: Ensuring the trained model remains completely unchanged without retraining."""

    def test_model_hash_unchanged(self):
        """Verify the SHA-256 hash of best_model.joblib matches the baseline exactly."""
        model_file = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_file.exists()

        with open(model_file, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        assert current_hash == BEST_MODEL_SHA256, (
            f"Trained model artifact was unexpectedly modified or retrained! "
            f"Expected: {BEST_MODEL_SHA256}, Got: {current_hash}"
        )


class TestGlobalFeatureImportanceRanking:
    """Test 11: Global importance calculations and ranking."""

    def test_global_importance_order_and_properties(self, preprocessed_shap_artifacts):
        """Verify global importance ranks lag and momentum features at the top."""
        artifacts = preprocessed_shap_artifacts
        global_df = compute_global_feature_importance(
            artifacts["shap_explanation"], artifacts["feature_names"]
        )

        assert len(global_df) == 12
        assert list(global_df["rank"]) == list(range(1, 13))

        # Mean absolute SHAP values must be strictly sorted descending
        mean_abs_values = global_df["mean_abs_shap"].tolist()
        assert mean_abs_values == sorted(mean_abs_values, reverse=True)
        assert all(v >= 0.0 for v in mean_abs_values)

        # The top 4 features should be temporal/lag features (lag_1, rolling_mean_2, pct_change, lag_2)
        top_4_raw = global_df["feature_raw"].head(4).tolist()
        assert "num__lag_1" in top_4_raw
        assert "num__rolling_mean_2" in top_4_raw
        assert "num__lag_2" in top_4_raw
        assert "num__historical_pct_change" in top_4_raw


class TestSavedArtifactsHealth:
    """Test 12: Verify all saved explainability artifacts exist and have valid content."""

    def test_saved_artifacts_exist_and_non_empty(self):
        """Verify CSV, JSON, PNG, and Markdown files in models/evaluation/shap/."""
        assert SHAP_DIR.exists()

        csv_file = SHAP_DIR / "shap_global_importance.csv"
        assert csv_file.exists()
        assert csv_file.stat().st_size > 500

        json_local = SHAP_DIR / "local_explanation_urea_2022_23.json"
        assert json_local.exists()
        with open(json_local, "r", encoding="utf-8") as f:
            local_data = json.load(f)
        assert local_data["commodity_name"] == "Urea"
        assert local_data["additivity_verified"] is True

        json_meta = SHAP_DIR / "shap_metadata.json"
        assert json_meta.exists()
        with open(json_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        assert meta_data["explainer_type"] == "shap.LinearExplainer"
        assert meta_data["additivity_verified_all_samples"] is True

        md_doc = SHAP_DIR / "shap_assumptions_and_limitations.md"
        assert md_doc.exists()
        assert md_doc.stat().st_size > 1000

        # Check plot images
        beeswarm_img = SHAP_DIR / "summary_beeswarm.png"
        assert beeswarm_img.exists()
        assert beeswarm_img.stat().st_size > 50000

        bar_img = SHAP_DIR / "summary_bar.png"
        assert bar_img.exists()
        assert bar_img.stat().st_size > 50000

        waterfall_img = SHAP_DIR / "waterfall_urea_2022_23.png"
        assert waterfall_img.exists()
        assert waterfall_img.stat().st_size > 50000
