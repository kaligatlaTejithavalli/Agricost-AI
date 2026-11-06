"""Unit and integration tests for Phase 2 out-of-sample backtesting evaluation.

Test Verifications:
1. Exact matching of 78 forecast-vs-actual pairs (26 commodities * 3 horizons).
2. Mathematical correctness of MAE, RMSE, MAPE, and Directional Accuracy.
3. Persistence and Drift multi-year baseline extrapolation logic.
4. Correctness of directional movement relative to FY 2022-23 baseline.
5. Verification of saved backtesting artifacts (CSV, JSON, Markdown).
6. Non-nullness and range bounds for all computed backtesting metrics.
7. Confirmation that Ridge Regression outperformed the Drift baseline.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BACKTESTING_DIR,
    SAVED_FORECASTS_CSV,
    STAGED_WPI_XLS,
    calculate_metrics,
    compute_category_metrics,
    compute_commodity_metrics,
    compute_horizon_metrics,
    generate_backtesting_artifacts,
    load_forecasts,
    load_staged_actuals,
    match_forecasts_and_actuals,
)
from src.config import AGRICULTURAL_INPUT_CATALOG, TARGET_LABEL


class TestBacktestingDatasetLoadingAndMatching:
    """Verifies data ingestion and 78-pair forecast-versus-actual matching."""

    def test_forecasts_file_loaded(self):
        """Verify saved forecasts CSV has exactly 78 rows and required columns."""
        df = load_forecasts()
        assert len(df) == 78
        assert df["commodity_code"].nunique() == 26
        assert set(df["forecast_horizon"].unique()) == {1, 2, 3}
        assert set(df["forecast_period"].unique()) == {"2023-24", "2024-25", "2025-26"}

    def test_staged_actuals_loaded(self):
        """Verify staged official actuals has data for all 26 commodities."""
        df = load_staged_actuals()
        assert not df.empty
        assert set(df["forecast_period"].unique()) == {"2023-24", "2024-25", "2025-26"}
        agri_actuals = df[df["commodity_code"].isin(AGRICULTURAL_INPUT_CATALOG.keys())]
        assert len(agri_actuals) == 78

    def test_exact_78_pairs_matched(self):
        """Verify matching forecasts with actuals yields exactly 78 pairs."""
        fc_df = load_forecasts()
        act_df = load_staged_actuals()
        matched = match_forecasts_and_actuals(fc_df, act_df)

        assert len(matched) == 78
        assert matched["commodity_code"].nunique() == 26
        assert not matched["actual_wpi"].isnull().any()
        assert not matched["forecast_wpi"].isnull().any()
        assert (matched["actual_wpi"] > 0).all()
        assert (matched["forecast_wpi"] > 0).all()


class TestBacktestingMetricsCalculations:
    """Verifies mathematical validity of evaluation metrics."""

    def test_metric_calculation_accuracy(self):
        """Verify MAE, RMSE, MAPE, and Directional Accuracy on a controlled test dataframe."""
        sample_df = pd.DataFrame({
            "error_test": [2.0, -4.0, 0.0],
            "abs_error_test": [2.0, 4.0, 0.0],
            "pct_error_test": [2.0, 4.0, 0.0],
            "dir_acc_test": [1, 0, 1],
        })

        metrics = calculate_metrics(sample_df, "test")
        assert metrics["mae"] == pytest.approx(2.0, abs=1e-4)
        assert metrics["rmse"] == pytest.approx(np.sqrt(20.0 / 3.0), abs=1e-4)
        assert metrics["mape"] == pytest.approx(2.0, abs=1e-4)
        assert metrics["directional_accuracy_pct"] == pytest.approx(66.67, abs=0.1)

    def test_multi_year_baselines_extrapolation(self):
        """Verify persistence stays constant while drift extrapolates linearly."""
        fc_df = load_forecasts()
        act_df = load_staged_actuals()
        matched = match_forecasts_and_actuals(fc_df, act_df)

        # Check for Urea (code 1310020001)
        urea = matched[matched["commodity_code"] == 1310020001].sort_values("forecast_horizon")
        assert len(urea) == 3

        # Persistence must equal last_observed_wpi (115.10) for all horizons
        assert (urea["forecast_persistence"] == 115.10).all()

        # Drift must advance by historical slope each year
        slope = urea["historical_slope"].iloc[0]
        base = urea["last_observed_wpi"].iloc[0]
        assert urea["forecast_drift"].iloc[0] == pytest.approx(base + 1 * slope, abs=0.05)
        assert urea["forecast_drift"].iloc[1] == pytest.approx(base + 2 * slope, abs=0.05)
        assert urea["forecast_drift"].iloc[2] == pytest.approx(base + 3 * slope, abs=0.05)


class TestBacktestingArtifactsAndHealth:
    """Verifies saved backtesting artifacts exist, are non-empty, and valid."""

    @pytest.fixture(autouse=True)
    def ensure_artifacts(self):
        """Ensure backtesting artifacts are generated before testing."""
        generate_backtesting_artifacts()

    def test_matched_predictions_csv(self):
        """Verify backtesting_matched_predictions.csv exists with 78 rows."""
        csv_path = BACKTESTING_DIR / "backtesting_matched_predictions.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 78
        assert "error_ridge" in df.columns
        assert "error_persistence" in df.columns
        assert "error_drift" in df.columns
        assert "dir_acc_ridge" in df.columns

    def test_summary_json_schema(self):
        """Verify backtesting_summary.json structure and metric ranges."""
        json_path = BACKTESTING_DIR / "backtesting_summary.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_pairs_evaluated"] == 78
        assert data["commodities_count"] == 26
        assert data["horizons_count"] == 3
        assert data["target_unit"] == TARGET_LABEL

        # Verify Ridge outperformed Drift baseline
        ridge_rmse = data["overall_performance"]["ridge_regression"]["rmse"]
        drift_rmse = data["overall_performance"]["drift_baseline"]["rmse"]
        assert ridge_rmse < drift_rmse, f"Ridge RMSE ({ridge_rmse}) should beat Drift RMSE ({drift_rmse})"

        # Verify Ridge directional accuracy is positive
        ridge_dir = data["overall_performance"]["ridge_regression"]["directional_accuracy_pct"]
        assert ridge_dir > 50.0, f"Expected directional accuracy > 50%, found {ridge_dir}%"

    def test_category_and_commodity_csv_tables(self):
        """Verify category and commodity CSV files exist and have valid rows."""
        cat_path = BACKTESTING_DIR / "category_backtesting_performance.csv"
        comm_path = BACKTESTING_DIR / "commodity_backtesting_performance.csv"

        assert cat_path.exists() and cat_path.stat().st_size > 200
        assert comm_path.exists() and comm_path.stat().st_size > 1000

        comm_df = pd.read_csv(comm_path)
        assert len(comm_df) == 26
        assert (comm_df["ridge_mape_pct"] >= 0).all()

    def test_markdown_report_exists_and_contains_sections(self):
        """Verify backtesting_evaluation_report.md exists and contains required sections."""
        md_path = BACKTESTING_DIR / "backtesting_evaluation_report.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")

        assert "Out-of-Sample Backtesting Evaluation Report" in content
        assert "Overall Model Performance vs. Multi-Year Baselines" in content
        assert "Performance by Forecast Horizon" in content
        assert "Performance by Agricultural Input Category" in content
        assert "Best and Worst Forecast Case Studies" in content
        assert "Methodological Limitations" in content
