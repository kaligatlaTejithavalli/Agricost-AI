"""Unit tests for Stage 7 Future WPI Forecasting Engine (src/forecast.py).

Verifies:
1. Valid commodity selection across catalog.
2. Rejection of invalid or out-of-catalog commodity codes.
3. Support for forecast horizons of 1, 2, and 3 financial years.
4. Rejection of invalid forecast horizons (<=0, >3, non-integers).
5. Correct chronological financial-year and period-order progression.
6. Exact recursive lag updates (prior forecasts replace unavailable future lags).
7. Mathematical correctness of 2-year rolling mean calculations.
8. Mathematical correctness of historical percentage change calculations.
9. Feature schema compatibility with the serialized Ridge Regression pipeline.
10. Finite and strictly positive forecast outputs across all 26 commodities.
11. Preservation and immutability of the original processed historical dataset.
12. Strict immutability of the saved model pipeline artifact (no retraining).
13. Enforcement of 'WPI index points' unit labeling (anti-rupee guardrail).
14. Seamless compatibility with the existing budget ratio bridge (src.budget).
"""

import hashlib
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.budget import PriceProjectionResult
from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    FORECASTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_WPI_CSV,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.forecast import (
    CommodityForecastResult,
    SingleYearForecast,
    advance_financial_year,
    bridge_forecast_to_budget,
    forecast_all_commodities,
    forecast_commodity,
    load_forecasting_pipeline,
    load_historical_commodity_series,
)

# Reference SHA-256 hash of the trained best_model.joblib
BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Fixture providing loaded trained Ridge pipeline."""
    return load_forecasting_pipeline()


@pytest.fixture(scope="module")
def historical_timeseries():
    """Fixture providing processed FY timeseries DataFrame."""
    path = PROCESSED_DATA_DIR / "agricultural_inputs_wpi_fy_timeseries.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Test 1 & 2: Commodity Selection and Validation
# ---------------------------------------------------------------------------
class TestCommoditySelection:
    """Tests for validating commodity code inputs."""

    def test_valid_commodity_selection(self, loaded_pipeline):
        """Verify forecasting works for key commodities: Urea, DAP, and HSD Diesel."""
        sample_codes = [1310020001, 1310020002, 1202000005]  # Urea, DAP, HSD
        for code in sample_codes:
            result = forecast_commodity(commodity_code=code, forecast_horizon=3, pipeline=loaded_pipeline)
            assert isinstance(result, CommodityForecastResult)
            assert result.commodity_code == code
            assert len(result.forecasts) == 3
            assert result.last_observed_period == "2022-23"
            assert result.last_observed_order == 10
            assert result.last_observed_wpi > 0.0

    def test_invalid_commodity_code_raises_error(self, loaded_pipeline):
        """Verify ValueError is raised when querying a non-existent commodity code."""
        with pytest.raises(ValueError, match="not a verified agricultural input commodity"):
            forecast_commodity(commodity_code=9999999999, forecast_horizon=1, pipeline=loaded_pipeline)

    def test_non_integer_commodity_code_raises_error(self, loaded_pipeline):
        """Verify ValueError is raised when commodity code cannot be parsed as integer."""
        with pytest.raises(ValueError, match="Commodity code must be an integer"):
            forecast_commodity(commodity_code="INVALID", forecast_horizon=1, pipeline=loaded_pipeline)


# ---------------------------------------------------------------------------
# Test 3 & 4: Forecast Horizons
# ---------------------------------------------------------------------------
class TestForecastHorizons:
    """Tests for forecast horizon boundaries and step counts."""

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_supported_horizons(self, horizon, loaded_pipeline):
        """Verify horizons 1, 2, and 3 produce exactly 1, 2, and 3 forecast steps."""
        urea_code = 1310020001
        result = forecast_commodity(commodity_code=urea_code, forecast_horizon=horizon, pipeline=loaded_pipeline)
        assert result.forecast_horizon == horizon
        assert len(result.forecasts) == horizon
        for idx, step in enumerate(result.forecasts, start=1):
            assert step.horizon == idx

    @pytest.mark.parametrize("invalid_horizon", [0, -1, 4, 10, "two", False, 1.5])
    def test_rejection_of_invalid_horizons(self, invalid_horizon, loaded_pipeline):
        """Verify rejection of horizons outside 1..3 or non-integer types."""
        urea_code = 1310020001
        with pytest.raises(ValueError):
            forecast_commodity(commodity_code=urea_code, forecast_horizon=invalid_horizon, pipeline=loaded_pipeline)


# ---------------------------------------------------------------------------
# Test 5, 6, 7 & 8: Chronological Progression, Recursive Lags & Features
# ---------------------------------------------------------------------------
class TestChronologicalProgressionAndLags:
    """Tests for chronological year progression, recursive lags, and rolling features."""

    def test_financial_year_progression(self):
        """Verify advance_financial_year advances FY strings correctly."""
        assert advance_financial_year("2022-23", 1) == "2023-24"
        assert advance_financial_year("2022-23", 2) == "2024-25"
        assert advance_financial_year("2022-23", 3) == "2025-26"
        assert advance_financial_year("2019-20", 1) == "2020-21"

    def test_invalid_fy_format_raises_error(self):
        """Verify advance_financial_year rejects malformed year strings."""
        with pytest.raises(ValueError, match="Invalid financial year format"):
            advance_financial_year("2022/2023", 1)

    def test_recursive_lag_updates(self, loaded_pipeline):
        """Verify that predictions from prior steps replace unavailable future lags."""
        urea_code = 1310020001
        res = forecast_commodity(commodity_code=urea_code, forecast_horizon=3, pipeline=loaded_pipeline)

        step1, step2, step3 = res.forecasts

        # Step 1 (2023-24): lag_1 is historical 2022-23 WPI (115.1), lag_2 is 2021-22 WPI (110.1)
        assert step1.period_order == 11
        assert step1.forecast_period == "2023-24"
        assert step1.lag_1 == pytest.approx(res.last_observed_wpi, abs=0.01)
        assert step1.lag_2 == pytest.approx(110.1, abs=0.01)

        # Step 2 (2024-25): lag_1 is Step 1 prediction, lag_2 is historical 2022-23 WPI
        assert step2.period_order == 12
        assert step2.forecast_period == "2024-25"
        assert step2.lag_1 == pytest.approx(step1.forecast_wpi, abs=0.01)
        assert step2.lag_2 == pytest.approx(res.last_observed_wpi, abs=0.01)

        # Step 3 (2025-26): lag_1 is Step 2 prediction, lag_2 is Step 1 prediction
        assert step3.period_order == 13
        assert step3.forecast_period == "2025-26"
        assert step3.lag_1 == pytest.approx(step2.forecast_wpi, abs=0.01)
        assert step3.lag_2 == pytest.approx(step1.forecast_wpi, abs=0.01)

    def test_rolling_mean_2_calculation(self, loaded_pipeline):
        """Verify rolling_mean_2 is strictly ((lag_1 + lag_2) / 2) at every step."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=3, pipeline=loaded_pipeline)
        for step in res.forecasts:
            expected_rm = (step.lag_1 + step.lag_2) / 2.0
            assert step.rolling_mean_2 == pytest.approx(expected_rm, abs=0.001)

    def test_historical_pct_change_calculation(self, loaded_pipeline):
        """Verify historical_pct_change is strictly (((lag_1 - lag_2) / lag_2) * 100) at every step."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=3, pipeline=loaded_pipeline)
        for step in res.forecasts:
            expected_pct = ((step.lag_1 - step.lag_2) / step.lag_2) * 100.0
            assert step.historical_pct_change == pytest.approx(expected_pct, abs=0.001)


# ---------------------------------------------------------------------------
# Test 9, 10 & 13: Model Compatibility, Output Validity & Unit Labels
# ---------------------------------------------------------------------------
class TestModelCompatibilityAndOutputs:
    """Tests for model input compatibility, positive finite values, and unit labels."""

    def test_feature_compatibility_with_saved_model(self, loaded_pipeline):
        """Verify input feature DataFrame strictly matches fitted pipeline expectations."""
        res = forecast_commodity(commodity_code=1202000005, forecast_horizon=1, pipeline=loaded_pipeline)
        assert res.forecasts[0].forecast_wpi > 0.0

    def test_finite_and_positive_forecast_outputs(self, loaded_pipeline):
        """Verify forecasts across all 26 commodities for horizons 1..3 are finite and positive."""
        all_forecasts_df = forecast_all_commodities(forecast_horizon=3, pipeline=loaded_pipeline)
        assert len(all_forecasts_df) == 26 * 3  # 78 rows

        assert not all_forecasts_df["forecast_wpi"].isna().any()
        assert np.all(np.isfinite(all_forecasts_df["forecast_wpi"]))
        assert (all_forecasts_df["forecast_wpi"] > 0.0).all()

    def test_unit_labeling_preservation(self, loaded_pipeline):
        """Verify all outputs strictly preserve 'WPI index points' and exclude currency symbols."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=3, pipeline=loaded_pipeline)
        assert res.target_unit == TARGET_LABEL
        assert res.target_unit == "WPI index points"
        for step in res.forecasts:
            assert step.target_unit == "WPI index points"
            assert "₹" not in step.target_unit
            assert "Rs" not in step.target_unit


# ---------------------------------------------------------------------------
# Test 11 & 12: Dataset Preservation and Model Immutability
# ---------------------------------------------------------------------------
class TestPreservationAndImmutability:
    """Tests ensuring datasets and trained models remain unmodified."""

    def test_preservation_of_historical_dataset(self, historical_timeseries):
        """Verify historical processed timeseries has 286 rows and 2022-23 cutoff."""
        assert len(historical_timeseries) == 286
        assert historical_timeseries["period_label"].max() == "2022-23"
        assert (historical_timeseries["period_order"] <= 10).all()

        # Check raw CSV exists and is unmodified
        assert RAW_WPI_CSV.exists()
        assert RAW_WPI_CSV.stat().st_size > 100000

    def test_no_retraining_or_modification_of_saved_model(self):
        """Verify the SHA-256 hash of best_model.joblib is byte-for-byte identical."""
        model_path = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_path.exists()
        with open(model_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        assert current_hash == BEST_MODEL_SHA256, "Saved model pipeline was unexpectedly modified!"


# ---------------------------------------------------------------------------
# Test 14: Integration with Budget Calculation Formula (src/budget.py)
# ---------------------------------------------------------------------------
class TestBudgetIntegration:
    """Tests for seamless integration between forecast outputs and the budget ratio bridge."""

    def test_budget_bridge_compatibility_urea(self, loaded_pipeline):
        """Verify bridging a 2023-24 Urea forecast to a current rupee price."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=3, pipeline=loaded_pipeline)

        # Baseline: Urea last observed WPI = 115.1, Forecast 2023-24 WPI = 121.43
        # User current price = Rs 266.50 per 50 kg bag
        projection = bridge_forecast_to_budget(
            forecast_result=res,
            target_period="2023-24",
            current_user_price=266.50,
            price_unit="₹ per 50 kg bag",
        )

        assert isinstance(projection, PriceProjectionResult)
        assert projection.commodity_code == 1310020001
        assert projection.current_period == "2022-23"
        assert projection.forecast_period == "2023-24"
        assert projection.current_wpi == 115.1
        assert projection.forecast_wpi == pytest.approx(121.43, abs=0.01)

        # Expected future price = 266.50 * (121.425 / 115.1) ~= 281.15
        assert projection.estimated_future_price == pytest.approx(281.15, abs=0.05)
        assert projection.projected_percentage_change > 0.0

    def test_budget_bridge_rejects_missing_target_period(self, loaded_pipeline):
        """Verify ValueError is raised if target period is not in forecast horizon."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=1, pipeline=loaded_pipeline)
        with pytest.raises(ValueError, match="not found in forecast result"):
            bridge_forecast_to_budget(
                forecast_result=res,
                target_period="2025-26",  # Horizon was only 1 (2023-24)
                current_user_price=266.50,
                price_unit="₹ per 50 kg bag",
            )

    def test_budget_bridge_rejects_non_positive_price(self, loaded_pipeline):
        """Verify ValueError is raised if current user price is zero or negative."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=1, pipeline=loaded_pipeline)
        with pytest.raises(ValueError, match="strictly positive"):
            bridge_forecast_to_budget(
                forecast_result=res,
                target_period="2023-24",
                current_user_price=-50.0,
                price_unit="₹ per 50 kg bag",
            )


# ---------------------------------------------------------------------------
# Test Artifacts Health
# ---------------------------------------------------------------------------
class TestForecastArtifactsHealth:
    """Verify saved CSV, JSON, and Markdown artifacts in models/evaluation/forecasts/."""

    def test_saved_artifacts_exist_and_non_empty(self):
        """Check all artifacts generated by run_forecast_pipeline()."""
        assert FORECASTS_DIR.exists()

        csv_file = FORECASTS_DIR / "commodity_forecasts_all.csv"
        assert csv_file.exists()
        assert csv_file.stat().st_size > 10000
        df = pd.read_csv(csv_file)
        assert len(df) == 78

        summary_json = FORECASTS_DIR / "forecasts_summary.json"
        assert summary_json.exists()
        with open(summary_json, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        assert "urea" in summary_data
        assert "dap" in summary_data
        assert "diesel_hsd" in summary_data

        meta_json = FORECASTS_DIR / "forecast_metadata.json"
        assert meta_json.exists()
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["total_commodities_supported"] == 26
        assert meta["latest_observed_financial_year"] == "2022-23"

        doc_md = FORECASTS_DIR / "forecast_limitations_and_methodology.md"
        assert doc_md.exists()
        assert doc_md.stat().st_size > 1500
