"""Unit and integration tests for the AgriCost-AI Streamlit Dashboard (app.py).

Test Verifications for Phase 3H Dashboard Integration:
1. Extended historical WPI dataset loading (364 rows across 26 commodities through FY 2025-26).
2. Agricultural input commodity catalog and category selection.
3. Forecast retrieval and integration with recursive forecasting engine for FY 2026-27 to 2028-29.
4. Forecast horizon validation (horizons 1, 2, 3 from FY 2025-26 cutoff).
5. Budget calculation integration and ratio bridge correctness.
6. Strict WPI-versus-rupee unit labeling (anti-rupee guardrail).
7. Active model selection: Ridge Regression (alpha=10.0) [Production Candidate Extended].
8. SHAP artifact loading from extended directory (global importance, summary plots, local waterfall for Urea FY 2025-26).
9. Dual-trajectory forecasting and budget scenario UI elements.
10. Preservation of legacy model (best_model.joblib) and historical evaluation artifacts.
11. End-to-end dashboard execution with Streamlit AppTest (zero unhandled exceptions).
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
from src.budget import PriceProjectionResult, estimate_future_input_price
from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    EXTENDED_FORECASTS_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_SHAP_DIR,
    FORECASTS_DIR,
    PROCESSED_DATA_DIR,
    SHAP_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.forecast import forecast_commodity, load_forecasting_pipeline

BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"
CANDIDATE_MODEL_SHA256 = "0102c1afe2e044c4e11e26bd37ee7720c97a9cd990792ca651f911e7ba5f3f13"


# ---------------------------------------------------------------------------
# Test 1: App Startup and Execution via AppTest
# ---------------------------------------------------------------------------
class TestAppStartupAndExecution:
    """Verifies that app.py imports and runs end-to-end via Streamlit AppTest."""

    def test_app_imports_cleanly(self):
        """Verify app module has required functions and constants."""
        assert hasattr(app, "main")
        assert hasattr(app, "get_historical_timeseries")
        assert hasattr(app, "get_model_pipeline")
        assert hasattr(app, "render_tab_price_insights")
        assert hasattr(app, "render_tab_cost_forecast")
        assert hasattr(app, "render_tab_budget_planner")
        assert hasattr(app, "render_tab_explainable_ai")

    def test_apptest_runs_without_exceptions(self):
        """Run app.py via Streamlit AppTest and verify zero unhandled exceptions."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        assert not at.exception, f"AppTest raised an unhandled exception: {at.exception}"

    def test_apptest_renders_four_tabs(self):
        """Verify that the dashboard renders the 4 core tabs."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()
        assert len(at.tabs) == 4
        tab_labels = [tab.label for tab in at.tabs]
        assert any("Price Insights" in label for label in tab_labels)
        assert any("Cost Forecast" in label for label in tab_labels)
        assert any("Budget Planner" in label for label in tab_labels)
        assert any("Explainable AI" in label for label in tab_labels)


# ---------------------------------------------------------------------------
# Test 2: Commodity and Category Selection
# ---------------------------------------------------------------------------
class TestCommoditySelection:
    """Verifies catalog coverage and category grouping."""

    def test_all_26_commodities_present(self):
        """Verify the catalog contains all 26 verified input commodities."""
        assert len(AGRICULTURAL_INPUT_CATALOG) == 26
        categories = {meta.category for meta in AGRICULTURAL_INPUT_CATALOG.values()}
        assert "Fertilizer" in categories
        assert "Fuel & Power" in categories
        assert "Farm Machinery" in categories
        assert "Livestock Input" in categories
        assert "Pesticide / Agrochemical" in categories

    def test_sidebar_catalog_filtering(self):
        """Verify filtering by category returns correct commodity subset."""
        fert_commodities = [
            meta for meta in AGRICULTURAL_INPUT_CATALOG.values() if meta.category == "Fertilizer"
        ]
        assert len(fert_commodities) >= 5
        assert any(c.name == "Urea" for c in fert_commodities)
        assert any(c.name == "Di ammonium phosphate" for c in fert_commodities)


# ---------------------------------------------------------------------------
# Test 3 & 4: Historical Data Loading & Cutoff Period
# ---------------------------------------------------------------------------
class TestHistoricalDataLoading:
    """Verifies extended historical timeseries loading and data cutoff in app."""

    def test_historical_timeseries_loaded(self):
        """Verify get_historical_timeseries() returns 364 valid extended rows through FY 2025-26."""
        df = app.get_historical_timeseries()
        assert not df.empty
        assert len(df) == 364
        assert df["commodity_code"].nunique() == 26

    def test_latest_observed_period_is_2025_26(self):
        """Verify the latest observed period across all commodities is FY 2025-26."""
        df = app.get_historical_timeseries()
        assert df["period_label"].max() == "2025-26"
        latest_rows = df[df["period_label"] == "2025-26"]
        assert len(latest_rows) == 26

    def test_legacy_timeseries_preserved(self):
        """Verify the legacy 286-row historical timeseries remains intact on disk."""
        legacy_path = PROCESSED_DATA_DIR / "agricultural_inputs_wpi_fy_timeseries.csv"
        assert legacy_path.exists()
        legacy_df = pd.read_csv(legacy_path)
        assert len(legacy_df) == 286
        assert legacy_df["period_label"].max() == "2022-23"


# ---------------------------------------------------------------------------
# Test 5 & 6: Forecast Engine Retrieval & Horizons
# ---------------------------------------------------------------------------
class TestForecastEngineIntegration:
    """Verifies forecast retrieval, recursive lag updates, and horizon bounds."""

    @pytest.mark.parametrize("horizon", [1, 2, 3])
    def test_forecast_retrieval_for_urea(self, horizon):
        """Verify forward forecast retrieval for Urea from FY 2025-26 cutoff across horizons 1, 2, 3."""
        pipe = app.get_model_pipeline()
        res = forecast_commodity(
            commodity_code=1310020001,
            forecast_horizon=horizon,
            pipeline=pipe,
            timeseries_path=EXTENDED_PROCESSED_TIMESERIES_CSV,
        )
        assert res.commodity_name == "Urea"
        assert len(res.forecasts) == horizon
        assert res.last_observed_period == "2025-26"
        assert res.last_observed_wpi == pytest.approx(115.6, abs=0.1)
        assert res.forecasts[0].forecast_period == "2026-27"
        assert res.forecasts[0].static_baseline_wpi == pytest.approx(115.6, abs=0.1)

    def test_precomputed_forecasts_loading(self):
        """Verify pre-generated extended forecasts CSV has 78 rows for FY 2026-27 to 2028-29."""
        df = app.get_precomputed_forecasts()
        assert not df.empty
        assert len(df) == 78
        assert df["commodity_code"].nunique() == 26
        assert set(df["forecast_period"].unique()) == {"2026-27", "2027-28", "2028-29"}
        assert (df["last_observed_period"] == "2025-26").all()

    def test_forecast_values_match_verified_artifacts(self):
        """Verify forecast values match verified Phase 3G artifacts for Urea and HSD."""
        df = app.get_precomputed_forecasts()
        urea = df[df["commodity_code"] == 1310020001].sort_values("forecast_horizon")
        assert len(urea) == 3
        assert urea["forecast_wpi"].iloc[0] == pytest.approx(122.96, abs=0.1)
        assert urea["forecast_wpi"].iloc[1] == pytest.approx(132.91, abs=0.1)
        assert urea["forecast_wpi"].iloc[2] == pytest.approx(142.91, abs=0.1)

        hsd = df[df["commodity_code"] == 1202000005].sort_values("forecast_horizon")
        assert len(hsd) == 3
        assert hsd["forecast_wpi"].iloc[0] == pytest.approx(160.48, abs=0.1)
        assert hsd["forecast_wpi"].iloc[1] == pytest.approx(161.36, abs=0.1)
        assert hsd["forecast_wpi"].iloc[2] == pytest.approx(162.90, abs=0.1)

    def test_forecast_horizon_labels_and_message(self):
        """Verify dynamically derived horizon labels and cutoff notice in Tab 2."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()

        # 1. Verify model projection notice references FY 2025-26
        info_texts = [info.value for info in at.info]
        assert any("2025-26" in text or "2025–26" in text for text in info_texts), (
            f"Expected '2025-26' in AppTest info blocks: {info_texts}"
        )


        # 2. Verify radio button labeled 'Forecast Horizon From Historical Data Cutoff'
        horizon_radios = [r for r in at.radio if "Forecast Horizon From Historical Data Cutoff" in r.label]
        assert len(horizon_radios) == 1, "Forecast Horizon radio group not found or mislabeled"
        horizon_radio = horizon_radios[0]

        # 3. Verify exact radio option strings for FY 2026-27 through 2028-29
        expected_options = [
            "1 Year After Data Cutoff (FY 2026-27)",
            "2 Years After Data Cutoff (FY 2027-28)",
            "3 Years After Data Cutoff (FY 2028-29)",
        ]
        assert horizon_radio.options == expected_options

        # 4. Verify 'Years Ahead' is strictly excluded
        for opt in horizon_radio.options:
            assert "Ahead" not in opt
            assert "Year Ahead" not in opt
            assert "Years Ahead" not in opt

    def test_legacy_forecast_utility_backward_compatibility(self):
        """Verify default forecast_commodity call without parameters retains backward compatibility."""
        legacy_pipe = load_forecasting_pipeline()
        res = forecast_commodity(commodity_code=1310020001, forecast_horizon=1, pipeline=legacy_pipe)
        assert res.last_observed_period == "2022-23"
        assert res.last_observed_wpi == pytest.approx(115.1, abs=0.1)
        assert res.forecasts[0].forecast_period == "2023-24"


# ---------------------------------------------------------------------------
# Test 7: Budget Planner Integration & Ratio Bridge
# ---------------------------------------------------------------------------
class TestBudgetPlannerIntegration:
    """Verifies budget estimation logic, ratio formula, and rupee units."""

    def test_budget_ratio_bridge_calculation_extended(self):
        """Verify future price = current_price * (forecast_wpi / current_wpi) for FY 2026-27."""
        current_wpi = 115.60  # Urea FY 2025-26
        forecast_wpi = 122.9643  # Urea FY 2026-27 (unrounded model forecast)
        current_user_price = 266.50  # Rs per 50 kg bag

        projection = estimate_future_input_price(
            current_wpi=current_wpi,
            forecast_wpi=forecast_wpi,
            current_user_price=current_user_price,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020001,
            commodity_name="Urea",
            current_period="2025-26",
            forecast_period="2026-27",
        )

        assert isinstance(projection, PriceProjectionResult)
        assert projection.estimated_future_price == 283.48
        assert projection.projected_percentage_change == pytest.approx(6.37, abs=0.05)

    def test_wpi_vs_rupee_unit_segregation(self):
        """Verify WPI values are never labeled as rupees."""
        projection = estimate_future_input_price(
            current_wpi=115.60,
            forecast_wpi=122.96,
            current_user_price=266.50,
            price_unit="₹ per 50 kg bag",
        )
        assert projection.target_unit == TARGET_LABEL
        assert projection.target_unit == "WPI index points"
        assert "₹" not in projection.target_unit

    def test_budget_planner_unit_formatting_in_apptest(self):
        """Verify Physical Purchase Unit field default and absence of duplicated rupee wording in UI."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()

        # 1. Verify Physical Purchase Unit input text
        unit_inputs = [ti for ti in at.text_input if "Physical Purchase Unit" in ti.label]
        assert len(unit_inputs) == 1, "Physical Purchase Unit input not found"
        unit_input = unit_inputs[0]
        # Must be clean physical unit ('50 kg bag'), NOT '₹ per 50 kg bag'
        assert unit_input.value == "50 kg bag", f"Expected '50 kg bag', got '{unit_input.value}'"
        assert "₹" not in unit_input.value

        # 2. Verify no duplicated rupee or rate prepositions in any text in the rendered dashboard
        all_markdown = [m.value for m in at.markdown]
        all_captions = [c.value for c in at.caption]
        all_texts = all_markdown + all_captions
        for text in all_texts:
            assert "per ₹ per" not in text, f"Duplicated rupee rate string found: {text}"
            assert "₹ per ₹ per" not in text, f"Duplicated rupee rate string found: {text}"
            assert "₹ per ₹" not in text, f"Duplicated rupee rate string found: {text}"


# ---------------------------------------------------------------------------
# Test 8: SHAP Artifact Loading & Explainability
# ---------------------------------------------------------------------------
class TestExplainabilityArtifactsLoading:
    """Verifies extended SHAP artifacts are loaded without recomputing on page refresh."""

    def test_shap_importance_table_loaded(self):
        """Verify extended global SHAP importance table contains 12 features."""
        df = app.get_shap_importance()
        assert not df.empty
        assert len(df) == 12
        assert "mean_abs_shap" in df.columns
        assert df["unit"].iloc[0] == "WPI index points"

    def test_local_urea_explanation_loaded(self):
        """Verify local explanation for Urea FY 2025-26 has verified additivity."""
        urea_dict = app.get_local_urea_explanation()
        assert urea_dict
        assert urea_dict.get("commodity_name") == "Urea"
        assert urea_dict.get("period_label") == "2025-26"
        assert urea_dict.get("additivity_verified") is True
        assert urea_dict.get("actual_wpi") == pytest.approx(115.6, abs=0.1)

    def test_shap_plot_images_exist(self):
        """Verify saved extended PNG plot files exist and have non-zero size."""
        beeswarm_img = EXTENDED_SHAP_DIR / "summary_beeswarm.png"
        bar_img = EXTENDED_SHAP_DIR / "summary_bar.png"
        waterfall_img = EXTENDED_SHAP_DIR / "waterfall_urea_2025_26.png"

        assert beeswarm_img.exists() and beeswarm_img.stat().st_size > 10000
        assert bar_img.exists() and bar_img.stat().st_size > 10000
        assert waterfall_img.exists() and waterfall_img.stat().st_size > 10000

    def test_legacy_shap_artifacts_preserved(self):
        """Verify original Stage 6 SHAP artifacts remain intact in models/evaluation/shap/."""
        orig_meta = SHAP_DIR / "shap_metadata.json"
        orig_waterfall = SHAP_DIR / "waterfall_urea_2022_23.png"
        assert orig_meta.exists()
        assert orig_waterfall.exists()


# ---------------------------------------------------------------------------
# Test 9: Evaluation Report & Model Benchmark Transparency
# ---------------------------------------------------------------------------
class TestEvaluationReportTransparency:
    """Verifies benchmark model reporting transparency."""

    def test_evaluation_report_loaded(self):
        """Verify report documents Ridge as best ML model and Drift as best overall."""
        report = app.get_evaluation_report()
        assert report
        assert "best_overall_method" in report
        assert report["best_overall_method"]["name"] == "Drift Baseline (Linear Momentum)"
        assert report["best_overall_method"]["rmse"] == 20.49

        assert "selected_ml_pipeline" in report
        assert report["selected_ml_pipeline"]["name"] == "Ridge Regression (alpha=10.0)"
        assert report["selected_ml_pipeline"]["rmse"] == 21.03

    def test_ridge_regression_model_parameter_formatting_and_unicode(self):
        """Verify Ridge Regression model parameter is formatted as 'Ridge Regression (α = 10.0)' in UI."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()

        # Check metric value
        metric_values = [str(m.value) for m in at.metric]
        assert "Ridge Regression (α = 10.0)" in metric_values, (
            f"Expected 'Ridge Regression (α = 10.0)' in metric values, found: {metric_values}"
        )

        # Check markdown blocks across main area and sidebar
        all_markdowns = [md.value for md in at.markdown] + [md.value for md in at.sidebar.markdown]
        
        # Verify occurrences of the formatted string
        matches = [m for m in all_markdowns if "Ridge Regression (α = 10.0)" in m]
        assert len(matches) >= 3, f"Expected at least 3 occurrences in markdown, found {len(matches)}"

        # Verify no corrupt/bell characters or broken escape sequences
        for m in all_markdowns:
            assert "\x07" not in m, "Found ASCII Bell character (\\x07) in user-facing markdown"
            assert "lpha=10.0" not in m, "Found unescaped or corrupted 'lpha=10.0' in markdown"
            assert "\ufffd" not in m, "Found Unicode replacement character (\\ufffd) in markdown"


# ---------------------------------------------------------------------------
# Test 10: Dual-Trajectory Forecasting & Backtesting Integration
# ---------------------------------------------------------------------------
class TestDualTrajectoryComparison:
    """Verifies dual-trajectory UI elements, comparison tables, and backtesting metrics."""

    def test_backtesting_data_loaders(self):
        """Verify backtesting loaders load valid metrics from backtesting artifacts."""
        comm_df = app.get_backtesting_commodity_metrics()
        assert not comm_df.empty, "Expected non-empty backtesting commodity dataframe"
        assert len(comm_df) == 26, f"Expected 26 commodities, found {len(comm_df)}"
        assert "ridge_mae_pts" in comm_df.columns
        assert "persistence_mae_pts" in comm_df.columns

        summary = app.get_backtesting_summary()
        assert summary, "Expected non-empty backtesting summary dictionary"
        assert "overall_performance" in summary
        assert "horizon_breakdown" in summary
        assert summary["total_pairs_evaluated"] == 78

    def test_dual_trajectory_ui_elements(self):
        """Verify AppTest renders dual-trajectory metrics, methodology, and comparison table."""
        at = AppTest.from_file("app.py", default_timeout=25)
        at.run()

        # Check metric labels for both trajectories
        metric_labels = [m.label for m in at.metric]
        assert any("Ridge ML" in label for label in metric_labels), (
            f"Expected 'Ridge ML' in metric labels: {metric_labels}"
        )
        assert any("Naïve Persistence" in label for label in metric_labels), (
            f"Expected 'Naïve Persistence' in metric labels: {metric_labels}"
        )

        # Check markdown blocks for dual-trajectory explanation and comparison table
        all_markdowns = [md.value for md in at.markdown]
        
        # 1. Methodology explanation
        assert any("Dual-Trajectory Forecasting Framework" in md for md in all_markdowns), (
            "Expected 'Dual-Trajectory Forecasting Framework' in app markdowns"
        )
        # 2. Comparison table header
        assert any("Dual-Trajectory Forecast Comparison Table" in md for md in all_markdowns), (
            "Expected 'Dual-Trajectory Forecast Comparison Table' in app markdowns"
        )
        # 3. Model distinction
        assert any("assumes the last observed WPI" in md or "unchanged into the future" in md for md in all_markdowns), (
            "Expected Naïve Persistence definition in markdown"
        )

    def test_budget_planner_preserves_forecasting_pipeline(self):
        """Verify Budget Planner uses the candidate pipeline model with extended timeseries."""
        pipe = app.get_model_pipeline()
        assert pipe is not None, "Pipeline must be loadable for budget planner"
        res = forecast_commodity(
            commodity_code=1310020001,
            forecast_horizon=1,
            pipeline=pipe,
            timeseries_path=EXTENDED_PROCESSED_TIMESERIES_CSV,
        )
        proj = estimate_future_input_price(
            current_wpi=res.last_observed_wpi,
            forecast_wpi=res.forecasts[0].forecast_wpi,
            current_user_price=266.50,
            price_unit="₹ per 50 kg bag",
        )
        assert proj.estimated_future_price == 283.48

    def test_both_trained_model_hashes_strictly_unmodified(self):
        """Verify deployed best_model.joblib and candidate model SHA-256 hashes are unmodified."""
        best_path = TRAINED_MODELS_DIR / "best_model.joblib"
        cand_path = TRAINED_MODELS_DIR / "production_candidate_extended.joblib"

        assert best_path.exists()
        assert cand_path.exists()

        assert hashlib.sha256(best_path.read_bytes()).hexdigest() == BEST_MODEL_SHA256
        assert hashlib.sha256(cand_path.read_bytes()).hexdigest() == CANDIDATE_MODEL_SHA256
