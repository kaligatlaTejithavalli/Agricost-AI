"""Automated Unit and Integration Tests for Dual-Trajectory Forecasting & Budgeting.

Verifies:
1. Dual forecast trajectories: ML Trend (Ridge) and Static Baseline.
2. Static baseline WPI remaining strictly constant across horizons 1, 2, and 3.
3. Correct WPI-to-rupee ratio calculations for both scenarios.
4. Static baseline scenario yielding future unit price equal to user price and 0% change.
5. End-to-end Streamlit dashboard execution with AppTest and dual-scenario verification in Budget Planner.
6. Full backward compatibility of legacy forecasting and budget functions.
7. Strict preservation of the deployed model artifact (best_model.joblib) SHA-256 hash.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

import app
from src.budget import (
    FarmBudgetPlan,
    FarmExpenditureItem,
    PriceProjectionResult,
    calculate_dual_scenario_expenditure,
    calculate_input_expenditure,
    estimate_dual_scenario_input_price,
    estimate_future_input_price,
)
from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.forecast import (
    CommodityForecastResult,
    SingleYearForecast,
    bridge_dual_forecast_to_budget,
    bridge_forecast_to_budget,
    forecast_commodity,
    load_forecasting_pipeline,
)

# Reference SHA-256 hash of the existing deployed best_model.joblib
BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"


@pytest.fixture(scope="module")
def loaded_pipeline():
    """Load the trained Ridge pipeline."""
    return load_forecasting_pipeline()


# ===========================================================================
# 1. Dual Forecast Trajectories Tests
# ===========================================================================
class TestDualForecastTrajectories:
    """Verifies that both forecast trajectories are generated and behave according to specifications."""

    def test_single_year_forecast_contains_both_trajectories(self, loaded_pipeline):
        """Verify SingleYearForecast encapsulates both forecast_wpi and static_baseline_wpi."""
        urea_code = 1310020001
        res = forecast_commodity(commodity_code=urea_code, forecast_horizon=3, pipeline=loaded_pipeline)

        for step in res.forecasts:
            assert hasattr(step, "forecast_wpi")
            assert hasattr(step, "static_baseline_wpi")
            assert step.forecast_wpi > 0.0
            assert step.static_baseline_wpi > 0.0

    def test_static_baseline_constant_across_all_horizons(self, loaded_pipeline):
        """Verify static_baseline_wpi is strictly constant across horizons and equals last_observed_wpi."""
        test_commodities = [1310020001, 1310020002, 1202000005, 1318090001]  # Urea, DAP, HSD, Tractor
        for code in test_commodities:
            res = forecast_commodity(commodity_code=code, forecast_horizon=3, pipeline=loaded_pipeline)
            last_obs = res.last_observed_wpi

            for step in res.forecasts:
                assert step.static_baseline_wpi == pytest.approx(last_obs, abs=1e-5), (
                    f"Static baseline for commodity {code} changed at horizon {step.horizon}!"
                )

    def test_dataframe_export_contains_dual_trajectories(self, loaded_pipeline):
        """Verify to_dataframe() includes both forecast_wpi and static_baseline_wpi columns."""
        res = forecast_commodity(commodity_code=1202000005, forecast_horizon=3, pipeline=loaded_pipeline)
        df = res.to_dataframe()

        assert "forecast_wpi" in df.columns
        assert "static_baseline_wpi" in df.columns
        assert len(df) == 3
        # Static baseline column must be constant
        assert df["static_baseline_wpi"].nunique() == 1
        assert df["static_baseline_wpi"].iloc[0] == pytest.approx(res.last_observed_wpi, abs=0.01)

    def test_to_dict_includes_static_baseline_wpi(self, loaded_pipeline):
        """Verify to_dict() serialization preserves static_baseline_wpi."""
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=1, pipeline=loaded_pipeline)
        d = res.forecasts[0].to_dict()
        assert "static_baseline_wpi" in d
        assert d["static_baseline_wpi"] == round(res.last_observed_wpi, 2)


# ===========================================================================
# 2. Dual-Scenario Budget Ratio Bridge Tests
# ===========================================================================
class TestDualScenarioBudgetBridge:
    """Verifies index-to-rupee calculations across both forecasting scenarios."""

    def test_estimate_dual_scenario_input_price(self):
        """Verify estimate_dual_scenario_input_price computes accurate prices for both scenarios."""
        current_wpi = 100.0
        forecast_wpi = 115.0  # +15% WPI inflation
        current_user_price = 500.0  # ₹500 per unit

        result = estimate_dual_scenario_input_price(
            current_wpi=current_wpi,
            forecast_wpi=forecast_wpi,
            current_user_price=current_user_price,
            price_unit="₹ per bag",
            commodity_code=1310020001,
            commodity_name="Urea",
            current_period="2022-23",
            forecast_period="2023-24",
        )

        assert "ml_trend" in result
        assert "static_baseline" in result

        ml_proj = result["ml_trend"]
        static_proj = result["static_baseline"]

        # ML Trend: 500 * (115 / 100) = 575.0
        assert ml_proj.estimated_future_price == 575.00
        assert ml_proj.projected_percentage_change == 15.00

        # Static Baseline: 500 * (100 / 100) = 500.0
        assert static_proj.estimated_future_price == 500.00
        assert static_proj.projected_percentage_change == 0.00
        assert static_proj.forecast_wpi == current_wpi

    def test_calculate_dual_scenario_expenditure(self):
        """Verify calculate_dual_scenario_expenditure computes farm expenditures for both scenarios."""
        current_wpi = 150.0
        forecast_wpi = 180.0  # +20%
        user_price = 100.0
        quantity = 25.0

        dual_prices = estimate_dual_scenario_input_price(
            current_wpi=current_wpi,
            forecast_wpi=forecast_wpi,
            current_user_price=user_price,
            price_unit="₹ per litre",
        )

        dual_exp = calculate_dual_scenario_expenditure(
            ml_trend_projection=dual_prices["ml_trend"],
            static_baseline_projection=dual_prices["static_baseline"],
            total_quantity=quantity,
        )

        ml_exp = dual_exp["ml_trend"]
        static_exp = dual_exp["static_baseline"]

        # ML Trend: Price = 120, Total = 3000
        assert ml_exp.estimated_future_total_cost == 3000.00
        assert ml_exp.current_total_cost == 2500.00
        assert ml_exp.estimated_cost_increase == 500.00

        # Static Baseline: Price = 100, Total = 2500
        assert static_exp.estimated_future_total_cost == 2500.00
        assert static_exp.current_total_cost == 2500.00
        assert static_exp.estimated_cost_increase == 0.00

    def test_bridge_dual_forecast_to_budget(self, loaded_pipeline):
        """Verify bridge_dual_forecast_to_budget bridges both trajectories from a CommodityForecastResult."""
        hsd_code = 1202000005
        res = forecast_commodity(commodity_code=hsd_code, forecast_horizon=2, pipeline=loaded_pipeline)

        projections = bridge_dual_forecast_to_budget(
            forecast_result=res,
            target_period="2024-25",
            current_user_price=95.0,
            price_unit="₹ per litre",
        )

        assert "ml_trend" in projections
        assert "static_baseline" in projections

        ml_proj = projections["ml_trend"]
        static_proj = projections["static_baseline"]

        assert ml_proj.commodity_code == hsd_code
        assert static_proj.commodity_code == hsd_code
        assert static_proj.estimated_future_price == 95.00
        assert static_proj.projected_percentage_change == 0.00
        assert static_proj.current_wpi == res.last_observed_wpi
        assert static_proj.forecast_wpi == res.last_observed_wpi


# ===========================================================================
# 3. Backward Compatibility Tests
# ===========================================================================
class TestBackwardCompatibility:
    """Ensures all existing public APIs continue working without modification."""

    def test_legacy_bridge_forecast_to_budget_unaffected(self, loaded_pipeline):
        """Verify existing bridge_forecast_to_budget continues returning PriceProjectionResult."""
        urea_code = 1310020001
        res = forecast_commodity(commodity_code=urea_code, forecast_horizon=1, pipeline=loaded_pipeline)

        proj = bridge_forecast_to_budget(
            forecast_result=res,
            target_period="2023-24",
            current_user_price=266.50,
            price_unit="₹ per 50 kg bag",
        )

        assert isinstance(proj, PriceProjectionResult)
        assert proj.commodity_code == urea_code
        assert proj.estimated_future_price == pytest.approx(281.15, abs=0.05)

    def test_legacy_estimate_future_input_price_unaffected(self):
        """Verify estimate_future_input_price maintains identical signature and output."""
        res = estimate_future_input_price(
            current_wpi=120.0,
            forecast_wpi=132.0,
            current_user_price=1000.0,
            price_unit="₹ per quintal",
        )
        assert isinstance(res, PriceProjectionResult)
        assert res.estimated_future_price == 1100.00
        assert res.projected_percentage_change == 10.00

    def test_legacy_calculate_input_expenditure_unaffected(self):
        """Verify calculate_input_expenditure maintains identical signature and output."""
        proj = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=110.0,
            current_user_price=50.0,
            price_unit="₹ per kg",
        )
        item = calculate_input_expenditure(
            price_projection=proj,
            total_quantity=100.0,
        )
        assert isinstance(item, FarmExpenditureItem)
        assert item.current_total_cost == 5000.00
        assert item.estimated_future_total_cost == 5500.00


# ===========================================================================
# 4. Streamlit App End-to-End Tests
# ===========================================================================
class TestAppDualScenarioExecution:
    """Verifies that app.py runs end-to-end via Streamlit AppTest with dual scenario components."""

    def test_apptest_runs_with_dual_scenario_budget(self):
        """Run app.py via AppTest and verify zero exceptions with dual scenario rendering."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        assert not at.exception, f"AppTest raised an exception: {at.exception}"

        # Verify Budget Planner has both scenario metrics
        metric_labels = [m.label for m in at.metric]
        assert any("Ridge ML Trend Budget" in label for label in metric_labels), (
            f"Expected 'Ridge ML Trend Budget' in metric labels: {metric_labels}"
        )
        assert any("Static Baseline Budget" in label for label in metric_labels), (
            f"Expected 'Static Baseline Budget' in metric labels: {metric_labels}"
        )
        assert any("Scenario Budget Divergence" in label for label in metric_labels), (
            f"Expected 'Scenario Budget Divergence' in metric labels: {metric_labels}"
        )

        # Verify side-by-side budget comparison table is rendered
        all_markdowns = [md.value for md in at.markdown]
        assert any("Side-by-Side Budget Scenario Comparison" in md for md in all_markdowns), (
            "Expected 'Side-by-Side Budget Scenario Comparison' in app markdowns"
        )
        assert any("Budget Scenario Disclaimer" in md for md in all_markdowns), (
            "Expected 'Budget Scenario Disclaimer' in app markdowns"
        )


# ===========================================================================
# 5. Deployed Model Artifact Preservation Test
# ===========================================================================
class TestModelArtifactPreservation:
    """Verifies that the deployed best_model.joblib is byte-for-byte pristine."""

    def test_deployed_model_hash_strictly_unmodified(self):
        """Verify best_model.joblib has exact reference SHA-256 hash."""
        model_file = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_file.exists(), "best_model.joblib not found!"
        with open(model_file, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        assert current_hash == BEST_MODEL_SHA256, (
            f"best_model.joblib SHA-256 mismatch! Expected {BEST_MODEL_SHA256}, got {current_hash}"
        )
