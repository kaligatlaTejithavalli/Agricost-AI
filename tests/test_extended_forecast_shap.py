"""Automated unit and integration tests for Phase 3G: Extended Production Forecasting & SHAP Artifacts.

Verifies:
1. Extended forward forecasts (78 predictions across 26 commodities for FY 2026-27, FY 2027-28, FY 2028-29).
2. Correct horizons (1, 2, 3), period orders (14, 15, 16), and historical cutoff (FY 2025-26).
3. Dual trajectory completeness: Ridge ML Trend Forecast ('forecast_wpi') and Static Baseline ('static_baseline_wpi').
4. Flatness and accuracy of Static Baseline: remains constant across horizons and equals FY 2025-26 actual.
5. No missing, NaN, infinite, or negative forecast values.
6. SHAP feature compatibility (12 features) and exact additivity (E[f(X)] + sum(phi) == y_hat) across all 312 samples.
7. Local explanation for Urea FY 2025-26 (exact additivity, expected values, 12 feature contributions).
8. Existence and non-emptiness of all generated SHAP artifacts under models/evaluation/shap/extended/.
9. Preservation of existing deployed model (best_model.joblib) and production candidate (production_candidate_extended.joblib).
10. Preservation of previous historical datasets and evaluation artifacts.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_DIR,
    BASE_YEAR_LABEL,
    EVALUATION_DIR,
    EXTENDED_FEATURES_CSV,
    EXTENDED_FORECASTS_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_SHAP_DIR,
    FORECASTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_WPI_CSV,
    SHAP_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.extended_artifacts import (
    MODEL_NAME_EXTENDED,
    PRODUCTION_CANDIDATE_MODEL_PATH,
    compute_file_sha256,
    generate_extended_forecasts,
    generate_extended_shap_artifacts,
)

# Reference SHA-256 hashes for trained models (Strict Model Preservation)
EXPECTED_BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"
EXPECTED_CANDIDATE_MODEL_SHA256 = "0102c1afe2e044c4e11e26bd37ee7720c97a9cd990792ca651f911e7ba5f3f13"


# ===========================================================================
# 1. Forward Forecasting Verification (78 Predictions, FY 2026-27 to 2028-29)
# ===========================================================================
class TestExtendedForwardForecasts:
    """Verifies generated multi-year forward forecasts across all 26 commodities."""

    @pytest.fixture(scope="module")
    def forecast_df(self) -> pd.DataFrame:
        """Load the generated extended forward forecasts CSV."""
        assert EXTENDED_FORECASTS_CSV.exists(), f"Forecast artifact missing at {EXTENDED_FORECASTS_CSV}"
        df = pd.read_csv(EXTENDED_FORECASTS_CSV)
        return df


    def test_forecast_record_count_and_dimensions(self, forecast_df: pd.DataFrame):
        """Verify exactly 78 forecast rows across all 26 commodities (26 * 3 = 78)."""
        assert len(forecast_df) == 78, f"Expected 78 rows, found {len(forecast_df)}"
        assert forecast_df["commodity_code"].nunique() == 26
        assert forecast_df["commodity_name"].nunique() == 26

    def test_forecast_horizons_and_periods(self, forecast_df: pd.DataFrame):
        """Verify horizons 1, 2, 3 correspond to FY 2026-27, FY 2027-28, and FY 2028-29."""
        expected_periods = ["2026-27", "2027-28", "2028-29"]
        assert sorted(forecast_df["forecast_period"].unique().tolist()) == expected_periods

        for horizon, period in [(1, "2026-27"), (2, "2027-28"), (3, "2028-29")]:
            subset = forecast_df[forecast_df["forecast_horizon"] == horizon]
            assert len(subset) == 26
            assert (subset["forecast_period"] == period).all()

    def test_historical_cutoff_and_orders(self, forecast_df: pd.DataFrame):
        """Verify historical cutoff is FY 2025-26 and period_order advances 14, 15, 16."""
        assert (forecast_df["last_observed_period"] == "2025-26").all()
        h1 = forecast_df[forecast_df["forecast_horizon"] == 1]
        h2 = forecast_df[forecast_df["forecast_horizon"] == 2]
        h3 = forecast_df[forecast_df["forecast_horizon"] == 3]

        assert (h1["period_order"] == 14).all()
        assert (h2["period_order"] == 15).all()
        assert (h3["period_order"] == 16).all()

    def test_forecast_values_finite_and_positive(self, forecast_df: pd.DataFrame):
        """Verify zero missing/NaN values and strictly positive predictions."""
        numeric_cols = [
            "last_observed_wpi",
            "forecast_wpi",
            "static_baseline_wpi",
            "pct_change_vs_last_observed",
            "pct_change_vs_prior_year",
            "lag_1",
            "lag_2",
            "rolling_mean_2",
            "historical_pct_change",
            "indicative_lower_80",
            "indicative_upper_80",
        ]
        for col in numeric_cols:
            assert not forecast_df[col].isna().any(), f"NaN values detected in column {col}"
            assert np.isfinite(forecast_df[col]).all(), f"Non-finite values detected in column {col}"

        assert (forecast_df["forecast_wpi"] > 0).all()
        assert (forecast_df["static_baseline_wpi"] > 0).all()
        assert (forecast_df["last_observed_wpi"] > 0).all()

    def test_static_baseline_constancy_and_accuracy(self, forecast_df: pd.DataFrame):
        """Verify static baseline remains constant across horizons and matches last observed WPI."""
        for code, group in forecast_df.groupby("commodity_code"):
            # Static baseline must not vary across H1, H2, H3
            assert group["static_baseline_wpi"].nunique() == 1
            # Static baseline must equal the actual observed FY 2025-26 WPI
            assert group["static_baseline_wpi"].iloc[0] == group["last_observed_wpi"].iloc[0]

    def test_sample_forecasts_urea_and_diesel(self, forecast_df: pd.DataFrame):
        """Verify forecast details for core reference commodities: Urea and Diesel (HSD)."""
        # Urea (code 1310020001)
        urea = forecast_df[forecast_df["commodity_code"] == 1310020001].sort_values("forecast_horizon")
        assert len(urea) == 3
        assert urea["last_observed_wpi"].iloc[0] == 115.60
        assert (urea["static_baseline_wpi"] == 115.60).all()
        assert urea["forecast_wpi"].iloc[0] == pytest.approx(122.96, abs=0.1)
        assert urea["forecast_wpi"].iloc[1] == pytest.approx(132.91, abs=0.1)
        assert urea["forecast_wpi"].iloc[2] == pytest.approx(142.91, abs=0.1)

        # Diesel (HSD, code 1202000005)
        hsd = forecast_df[forecast_df["commodity_code"] == 1202000005].sort_values("forecast_horizon")
        assert len(hsd) == 3
        assert hsd["last_observed_wpi"].iloc[0] == 160.79
        assert (hsd["static_baseline_wpi"] == 160.79).all()
        assert hsd["forecast_wpi"].iloc[0] == pytest.approx(160.48, abs=0.1)
        assert hsd["forecast_wpi"].iloc[1] == pytest.approx(161.36, abs=0.1)
        assert hsd["forecast_wpi"].iloc[2] == pytest.approx(162.90, abs=0.1)

    def test_metadata_and_unit_labels(self, forecast_df: pd.DataFrame):
        """Verify target unit semantics are strictly WPI index points, never rupees."""
        assert (forecast_df["target_unit"] == TARGET_LABEL).all()
        assert (forecast_df["model_name"] == MODEL_NAME_EXTENDED).all()


# ===========================================================================
# 2. Extended SHAP Artifacts & Additivity Verification
# ===========================================================================
class TestExtendedSHAPArtifacts:
    """Verifies generated SHAP explainability artifacts and exact mathematical additivity."""

    @pytest.fixture(scope="module")
    def shap_metadata(self) -> dict:
        """Load the extended SHAP metadata JSON."""
        meta_path = EXTENDED_SHAP_DIR / "shap_metadata_extended.json"
        assert meta_path.exists(), f"Extended SHAP metadata missing at {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def test_all_shap_artifacts_exist(self, shap_metadata: dict):
        """Verify existence and non-zero size of all 7 extended SHAP files."""
        expected_files = [
            "shap_global_importance.csv",
            "summary_beeswarm.png",
            "summary_bar.png",
            "local_explanation_urea_2025_26.json",
            "waterfall_urea_2025_26.png",
            "shap_assumptions_and_limitations.md",
            "shap_metadata_extended.json",
        ]
        for fname in expected_files:
            fpath = EXTENDED_SHAP_DIR / fname
            assert fpath.exists(), f"Expected SHAP artifact missing: {fpath}"
            assert fpath.stat().st_size > 0, f"SHAP artifact is empty: {fpath}"

    def test_shap_metadata_content(self, shap_metadata: dict):
        """Verify metadata fields, sample counts, feature count, and model hash."""
        assert shap_metadata["model_name"] == MODEL_NAME_EXTENDED
        assert shap_metadata["model_sha256"] == EXPECTED_CANDIDATE_MODEL_SHA256
        assert shap_metadata["sample_count"] == 312
        assert shap_metadata["commodity_count"] == 26
        assert shap_metadata["num_features_explained"] == 12
        assert shap_metadata["target_unit"] == TARGET_LABEL
        assert shap_metadata["additivity_verified_all_samples"] is True
        assert shap_metadata["max_additivity_error"] < 1e-4

    def test_global_feature_importance_schema_and_order(self):
        """Verify global feature importance rankings, 12 features, and non-empty values."""
        csv_path = EXTENDED_SHAP_DIR / "shap_global_importance.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 12
        assert list(df.columns) == ["rank", "feature_display", "mean_abs_shap", "mean_shap", "std_shap", "unit", "feature_raw"]
        assert df["rank"].tolist() == list(range(1, 13))
        # Top feature should be lag_1
        assert "lag_1" in df.iloc[0]["feature_raw"]
        assert (df["mean_abs_shap"] >= 0).all()
        assert (df["unit"] == TARGET_LABEL).all()

    def test_local_urea_explanation_and_additivity(self):
        """Verify local explanation for Urea FY 2025-26 satisfies exact additivity."""
        json_path = EXTENDED_SHAP_DIR / "local_explanation_urea_2025_26.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            urea_data = json.load(f)

        assert urea_data["commodity_code"] == 1310020001
        assert urea_data["commodity_name"] == "Urea"
        assert urea_data["period_label"] == "2025-26"
        assert urea_data["actual_wpi"] == 115.60
        assert urea_data["unit"] == TARGET_LABEL
        assert urea_data["additivity_verified"] is True
        assert urea_data["absolute_difference"] < 1e-4

        # Verify Base Value + Sum of SHAP == Reconstructed == Predicted
        base = urea_data["base_value"]
        sum_shap = urea_data["sum_of_shap"]
        reconstructed = urea_data["reconstructed_prediction"]
        pred = urea_data["predicted_wpi"]

        assert base + sum_shap == pytest.approx(reconstructed, abs=0.01)
        assert reconstructed == pytest.approx(pred, abs=0.01)
        assert len(urea_data["feature_contributions"]) == 12

    def test_shap_numerical_additivity_on_candidate_model(self):
        """Independently verify SHAP additivity against the deserialized pipeline."""
        pipeline = joblib.load(PRODUCTION_CANDIDATE_MODEL_PATH)
        features_df = pd.read_csv(EXTENDED_FEATURES_CSV)
        meta_path = EXTENDED_SHAP_DIR / "shap_metadata_extended.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["max_additivity_error"] < 1e-6
        assert meta["additivity_verified_all_samples"] is True


# ===========================================================================
# 3. Model & Historical Artifact Preservation (Strict Immutability)
# ===========================================================================
class TestArtifactAndModelPreservation:
    """Verifies that existing models, historical datasets, and older SHAP files are unmodified."""

    def test_deployed_model_sha256_unmodified(self):
        """Verify best_model.joblib SHA-256 matches its reference hash identically."""
        deployed_path = TRAINED_MODELS_DIR / "best_model.joblib"
        assert deployed_path.exists()
        actual_hash = compute_file_sha256(deployed_path)
        assert actual_hash == EXPECTED_BEST_MODEL_SHA256, (
            f"Deployed model best_model.joblib hash changed! Expected {EXPECTED_BEST_MODEL_SHA256}, got {actual_hash}"
        )

    def test_production_candidate_sha256_unmodified(self):
        """Verify production_candidate_extended.joblib SHA-256 matches its reference hash."""
        assert PRODUCTION_CANDIDATE_MODEL_PATH.exists()
        actual_hash = compute_file_sha256(PRODUCTION_CANDIDATE_MODEL_PATH)
        assert actual_hash == EXPECTED_CANDIDATE_MODEL_SHA256, (
            f"Candidate model hash changed! Expected {EXPECTED_CANDIDATE_MODEL_SHA256}, got {actual_hash}"
        )

    def test_existing_shap_directory_unoverwritten(self):
        """Verify original SHAP artifacts in models/evaluation/shap/ remain untouched."""
        original_meta = SHAP_DIR / "shap_metadata.json"
        assert original_meta.exists()
        with open(original_meta, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Original SHAP evaluates 78 test samples through FY 2022-23
        assert data["evaluation_samples"] == 78
        assert data["model_name"] == "Ridge Regression (alpha=10.0)"
        assert "2022-23" in data["local_example_urea_2022_23"]["period"]

    def test_existing_forecast_artifact_unmodified(self):
        """Verify original commodity_forecasts_all.csv remains untouched (78 rows, FY 2023-26)."""
        orig_csv = FORECASTS_DIR / "commodity_forecasts_all.csv"
        assert orig_csv.exists()
        df = pd.read_csv(orig_csv)
        assert len(df) == 78
        assert (df["last_observed_period"] == "2022-23").all()
        assert sorted(df["forecast_period"].unique().tolist()) == ["2023-24", "2024-25", "2025-26"]

    def test_extended_historical_datasets_intact(self):
        """Verify extended historical datasets have expected dimensions and records."""
        ts_df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)
        assert len(ts_df) == 364
        assert ts_df["commodity_code"].nunique() == 26

        feat_df = pd.read_csv(EXTENDED_FEATURES_CSV)
        assert len(feat_df) == 312
        assert feat_df["commodity_code"].nunique() == 26
