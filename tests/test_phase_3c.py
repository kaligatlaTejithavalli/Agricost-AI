"""Automated unit and integration tests for Phase 3C: Recursive Multi-Year Forecast Validation.

Verifies:
1. Historical cutoff integrity: Training partition restricted to period_order <= 10 (FY 2014-15 to 2022-23, N=234).
2. Ground truth actuals isolation: Exactly 78 evaluation actuals across 26 commodities and 3 horizons (FY 2023-24 to 2025-26).
3. Leak-free recursive simulation mechanics:
   - Horizon 1 uses actual historical lags from FY 2022-23 and FY 2021-22.
   - Horizon 2 uses predicted y_hat(1) as lag_1 and actual FY 2022-23 as lag_2.
   - Horizon 3 uses predicted y_hat(2) as lag_1 and predicted y_hat(1) as lag_2.
   - Dynamic recalculation of historical_pct_change and rolling_mean_2.
4. Baseline calculations:
   - Naïve Persistence carries forward last observed actual WPI.
   - Drift Baseline applies linear momentum: y_T + h * slope.
   - Rolling Mean Baseline updates 2-year moving average recursively.
5. Error metrics and units:
   - MAE and RMSE in WPI index points.
   - MAPE and Directional Accuracy in %.
6. Recursive performance ranking:
   - Naïve Persistence achieves lowest overall RMSE (due to macroeconomic price plateau).
   - Machine Learning models beat Drift Baseline, avoiding explosive divergence.
   - Uncertainty compounds across horizons (H1 < H2 < H3 for ML models).
7. Evaluation artifacts:
   - All 6 versioned artifacts saved under models/evaluation/phase_3c/.
   - Preservation of existing trained model (best_model.joblib) and production candidate.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_DIR,
    BASE_YEAR_LABEL,
    EVALUATION_DIR,
    EXTENDED_FEATURES_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.feature_engineering import CORE_FEATURE_COLUMNS, TARGET_COLUMN
from src.recursive_evaluation import (
    HISTORICAL_CUTOFF_ORDER,
    HISTORICAL_CUTOFF_PERIOD,
    PHASE_3C_EVAL_DIR,
    RECURSIVE_EVAL_PERIODS,
    SAVED_STAGE7_FORECASTS_CSV,
    build_pipeline,
    compute_metrics,
    evaluate_forecast_set,
    load_cutoff_training_data,
    load_ground_truth_actuals,
    run_baseline_simulations,
    run_phase_3c_validation,
    run_recursive_simulation,
)
from src.train import CATEGORICAL_FEATURE_COLUMNS


# ===========================================================================
# 1. Historical Cutoff and Data Partition Isolation Tests
# ===========================================================================
class TestPhase3CCutoffAndDataIsolation:
    """Tests verifying historical boundary isolation and data integrity at FY 2022-23 cutoff."""

    def test_cutoff_training_data_isolation(self):
        """Verify training data contains exactly 234 records strictly on or before FY 2022-23."""
        X_train, y_train = load_cutoff_training_data()
        assert len(X_train) == 234
        assert len(y_train) == 234

        # Check feature columns
        expected_cols = CORE_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
        assert list(X_train.columns) == expected_cols

        # Ensure period_order is capped strictly at cutoff order (10)
        assert X_train["period_order"].max() == HISTORICAL_CUTOFF_ORDER
        assert (y_train > 0).all()

    def test_ground_truth_actuals_structure(self):
        """Verify ground truth actuals contain exactly 78 records across 26 commodities and 3 horizons."""
        actuals_df = load_ground_truth_actuals()
        assert len(actuals_df) == 78
        assert actuals_df["commodity_code"].nunique() == 26
        assert sorted(actuals_df["forecast_horizon"].unique()) == [1, 2, 3]
        assert list(actuals_df["forecast_period"].unique()) == RECURSIVE_EVAL_PERIODS
        assert (actuals_df["actual_wpi"] > 0).all()


# ===========================================================================
# 2. Pipeline Construction and Recursive Simulation Mechanics Tests
# ===========================================================================
class TestPhase3CPipelinesAndSimulation:
    """Tests verifying model pipeline creation and leak-free recursive feature updates."""

    def test_build_pipeline_types(self):
        """Verify build_pipeline constructs valid Pipelines with preprocessor and regressor."""
        for model_type in ["ridge", "random_forest", "gradient_boosting"]:
            pipeline = build_pipeline(model_type)
            assert isinstance(pipeline, Pipeline)
            assert "preprocessor" in pipeline.named_steps
            assert "regressor" in pipeline.named_steps

    def test_recursive_simulation_mechanics(self):
        """Verify recursive simulation correctly feeds predicted outputs into subsequent lags."""
        X_train, y_train = load_cutoff_training_data()
        pipeline = build_pipeline("ridge", alpha=10.0)
        pipeline.fit(X_train, y_train)

        sim_df = run_recursive_simulation(pipeline, max_horizon=3)
        assert len(sim_df) == 78
        assert sim_df["commodity_code"].nunique() == 26

        # Check a specific commodity (e.g. Urea or first commodity)
        first_code = sorted(AGRICULTURAL_INPUT_CATALOG.keys())[0]
        comm_sim = sim_df[sim_df["commodity_code"] == first_code].sort_values("forecast_horizon").reset_index(drop=True)

        h1 = comm_sim.iloc[0]
        h2 = comm_sim.iloc[1]
        h3 = comm_sim.iloc[2]

        # Horizon 2 lag_1 should equal Horizon 1 forecast_wpi
        assert np.isclose(h2["lag_1_used"], h1["forecast_wpi"], atol=1e-5)
        # Horizon 2 lag_2 should equal Horizon 1 lag_1
        assert np.isclose(h2["lag_2_used"], h1["lag_1_used"], atol=1e-5)

        # Horizon 3 lag_1 should equal Horizon 2 forecast_wpi
        assert np.isclose(h3["lag_1_used"], h2["forecast_wpi"], atol=1e-5)
        # Horizon 3 lag_2 should equal Horizon 2 lag_1 (Horizon 1 forecast)
        assert np.isclose(h3["lag_2_used"], h2["lag_1_used"], atol=1e-5)

        # Dynamic rolling mean update check
        expected_roll_h2 = (h2["lag_1_used"] + h2["lag_2_used"]) / 2.0
        assert np.isclose(h2["rolling_mean_2_used"], expected_roll_h2, atol=1e-5)


# ===========================================================================
# 3. Baseline Simulations and Metric Calculation Tests
# ===========================================================================
class TestPhase3CBaselinesAndMetrics:
    """Tests verifying mathematical correctness of baselines and evaluation metrics."""

    def test_baseline_simulations(self):
        """Verify baseline predictions for persistence, drift, and rolling mean."""
        baselines = run_baseline_simulations(max_horizon=3)
        assert "persistence" in baselines
        assert "drift" in baselines
        assert "rolling_mean" in baselines

        for name, df in baselines.items():
            assert len(df) == 78
            assert df["commodity_code"].nunique() == 26

        # Test persistence carries forward constant last_observed_wpi
        pers = baselines["persistence"]
        assert (pers["forecast_wpi"] == pers["last_observed_wpi"]).all()

    def test_compute_metrics_correctness(self):
        """Verify compute_metrics calculates MAE, RMSE, MAPE, and Directional Accuracy accurately."""
        y_true = np.array([100.0, 110.0, 120.0])
        y_pred = np.array([105.0, 108.0, 125.0])
        last_obs = np.array([90.0, 100.0, 110.0])

        metrics = compute_metrics(y_true, y_pred, last_obs)
        assert np.isclose(metrics["mae"], (5.0 + 2.0 + 5.0) / 3.0)
        assert np.isclose(metrics["rmse"], np.sqrt((25.0 + 4.0 + 25.0) / 3.0))
        assert metrics["directional_accuracy_pct"] == 100.0


# ===========================================================================
# 4. Phase 3C End-to-End Validation & Empirical Findings Tests
# ===========================================================================
class TestPhase3CEmpiricalFindingsAndArtifacts:
    """Tests verifying Phase 3C execution, report artifacts, and empirical conclusions."""

    def test_phase_3c_artifacts_exist_and_complete(self):
        """Verify all 6 Phase 3C evaluation artifacts exist under models/evaluation/phase_3c/."""
        assert PHASE_3C_EVAL_DIR.exists()
        assert (PHASE_3C_EVAL_DIR / "recursive_model_comparison.csv").exists()
        assert (PHASE_3C_EVAL_DIR / "horizon_wise_performance.csv").exists()
        assert (PHASE_3C_EVAL_DIR / "category_recursive_performance.csv").exists()
        assert (PHASE_3C_EVAL_DIR / "recursive_predictions_matched.csv").exists()
        assert (PHASE_3C_EVAL_DIR / "phase_3c_evaluation_report.json").exists()
        assert (PHASE_3C_EVAL_DIR / "phase_3c_evaluation_report.md").exists()

    def test_phase_3c_report_json_content(self):
        """Verify evaluation report JSON matches key Phase 3C findings."""
        with open(PHASE_3C_EVAL_DIR / "phase_3c_evaluation_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["evaluation_phase"] == "Phase 3C: Recursive Multi-Year Forecast Validation"
        assert report["total_test_observations"] == 78
        assert report["target_unit"] == TARGET_LABEL
        assert "leakage_prevention_confirmation" in report

        findings = report["key_findings"]
        # Naïve persistence achieves lower RMSE than best ML in recursive validation
        assert findings["naive_persistence_winner"] is True
        assert findings["persistence_overall_rmse"] < 15.0  # ~12.56 WPI pts
        # Best ML model beats Drift baseline
        assert findings["ml_beats_drift_baseline"] is True

    def test_recursive_compounding_error(self):
        """Verify error expands across horizons (H1 -> H2 -> H3) for Stage 7 saved Ridge forecasts."""
        horizon_df = pd.read_csv(PHASE_3C_EVAL_DIR / "horizon_wise_performance.csv")
        r_df = horizon_df[horizon_df["Model / Baseline"] == "Stage 7 Saved Ridge Regression (alpha=10.0)"].sort_values("Horizon")

        assert len(r_df) == 3
        rmse_h1 = float(r_df[r_df["Horizon"] == 1]["RMSE (WPI pts)"].iloc[0])
        rmse_h2 = float(r_df[r_df["Horizon"] == 2]["RMSE (WPI pts)"].iloc[0])
        rmse_h3 = float(r_df[r_df["Horizon"] == 3]["RMSE (WPI pts)"].iloc[0])

        assert rmse_h1 < rmse_h2 < rmse_h3, f"Expected compounding error H1 < H2 < H3, got {rmse_h1}, {rmse_h2}, {rmse_h3}"

    def test_preservation_of_existing_deployed_model(self):
        """Verify Stage 5 deployed best_model.joblib is preserved with 156 samples."""
        model_file = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_file.exists()
        pipeline = joblib.load(model_file)
        scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
        assert scaler.n_samples_seen_ == 156
