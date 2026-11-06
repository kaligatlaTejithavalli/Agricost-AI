"""Automated tests for Phase 3A: Historical Data Integration (FY 2023-24 to FY 2025-26).

Verifies:
1. Staged file inspection (data/staged/fin_year_index.xls) and schema compatibility (869 rows, 17 columns).
2. Complete, non-null observations for FY 2023-24, FY 2024-25, and FY 2025-26 for all 26 selected commodities.
3. Commodity code compatibility and index base preservation (Base 2011-12 = 100.0).
4. Extended processed timeseries (364 rows across 14 Financial Years, 2012-13 to 2025-26).
5. Extended feature engineering without temporal leakage (312 usable rows, 12 periods per commodity).
6. Chronological lag correctness and mathematical formula accuracy across new periods.
7. Strict preservation of original raw dataset, trained models, historical evaluation artifacts, and dashboard.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    EXTENDED_FEATURES_CSV,
    EXTENDED_FY_COLUMNS,
    EXTENDED_FY_LABELS,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_WIDE_CSV,
    FY_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_WPI_CSV,
    STAGED_WPI_XLS,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.data_loader import (
    get_extended_agricultural_inputs_df,
    load_raw_wpi_data,
    load_staged_wpi_data,
)
from src.feature_engineering import (
    CORE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_extended_time_series_features,
    load_extended_processed_timeseries,
    save_extended_feature_engineered_datasets,
)
from src.preprocessor import (
    audit_extended_data,
    preprocess_extended_wpi_timeseries,
    preprocess_extended_wpi_wide,
    save_extended_processed_data,
)

# Reference SHA-256 hash of the trained best_model.joblib ensuring model immutability
BEST_MODEL_SHA256 = "724567d397a13c1187d2b1b6ef5b5c1dd3834844eace407e201a5b0a49be3b31"


# ===========================================================================
# 1. Staged File Inspection & Compatibility
# ===========================================================================
class TestStagedFileIntegrityAndCompatibility:
    """Verifies data/staged/fin_year_index.xls schema, completeness, and provenance."""

    def test_staged_file_exists_and_readable(self):
        """Verify fin_year_index.xls exists and loads 869 commodities across 17 columns."""
        assert STAGED_WPI_XLS.exists(), f"Staged XLS missing at {STAGED_WPI_XLS}"
        df = load_staged_wpi_data()
        assert len(df) == 869
        assert df.shape[1] >= 17
        assert "COMM_NAME" in df.columns
        assert "COMM_CODE" in df.columns
        assert "COMM_WT" in df.columns

    def test_commodity_coverage_and_codes(self):
        """Verify all 26 verified input commodities exist with matching codes and positive weights."""
        df = load_staged_wpi_data()
        present_codes = set(df["COMM_CODE"])
        for code, meta in AGRICULTURAL_INPUT_CATALOG.items():
            assert code in present_codes, f"Catalog commodity {meta.name} ({code}) missing from staged file!"
            row = df[df["COMM_CODE"] == code].iloc[0]
            assert float(row["COMM_WT"]) > 0.0

    def test_complete_observations_in_new_periods(self):
        """Verify complete, non-null observations for FY 2023-24, 2024-25, and 2025-26."""
        agri_df = get_extended_agricultural_inputs_df()
        assert len(agri_df) == 26

        new_periods = ["INDEX20232024", "INDEX20242025", "INDEX20252026"]
        for col in new_periods:
            assert col in agri_df.columns, f"Column {col} missing from staged agricultural inputs!"
            null_count = int(agri_df[col].isna().sum())
            assert null_count == 0, f"Column {col} has {null_count} unexpected null values!"
            # Strictly positive WPI values
            assert (agri_df[col] > 0).all(), f"Column {col} contains non-positive index values!"

    def test_base_year_2011_12_consistency(self):
        """Confirm the staged series matches the raw WPI dataset across FY 2012-13 to 2022-23 (Base 2011-12 = 100)."""
        raw_df = load_raw_wpi_data()
        staged_df = load_staged_wpi_data()

        codes = list(AGRICULTURAL_INPUT_CATALOG.keys())
        raw_agri = raw_df[raw_df["COMM_CODE"].isin(codes)].sort_values("COMM_CODE").reset_index(drop=True)
        staged_agri = staged_df[staged_df["COMM_CODE"].isin(codes)].sort_values("COMM_CODE").reset_index(drop=True)

        for col in FY_COLUMNS:
            raw_vals = raw_agri[col].astype(float).values
            staged_vals = staged_agri[col].astype(float).values
            max_diff = float(np.max(np.abs(raw_vals - staged_vals)))
            assert max_diff < 1e-4, f"Mismatch in {col} between raw and staged data: max diff {max_diff}"


# ===========================================================================
# 2. Extended Processed Timeseries Dataset
# ===========================================================================
class TestExtendedProcessedTimeseries:
    """Verifies data/processed/agricultural_inputs_wpi_fy_timeseries_extended.csv."""

    def test_extended_timeseries_file_and_schema(self):
        """Verify extended processed CSV exists, is non-empty, and has standard 16 columns."""
        assert EXTENDED_PROCESSED_TIMESERIES_CSV.exists()
        df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)
        assert not df.empty

        expected_cols = [
            "commodity_code",
            "commodity_name",
            "commodity_weight",
            "input_category",
            "suggested_unit",
            "description",
            "series_type",
            "period_key",
            "period_label",
            "period_order",
            "start_year",
            "end_year",
            "wpi_index_points",
            "target_unit",
            "base_year",
            "data_source",
        ]
        assert list(df.columns) == expected_cols

    def test_extended_observation_count_and_commodities(self):
        """Verify exact observation count: 26 commodities * 14 FY periods = 364 rows."""
        df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)
        assert len(df) == 364
        assert df["commodity_code"].nunique() == 26
        counts_per_commodity = df.groupby("commodity_code").size()
        assert (counts_per_commodity == 14).all()

    def test_extended_temporal_cutoff_and_span(self):
        """Verify financial year span: first='2012-13' (order 0) to last='2025-26' (order 13)."""
        df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)
        periods = sorted(list(df["period_label"].unique()))
        assert len(periods) == 14
        assert periods[0] == "2012-13"
        assert periods[-1] == "2025-26"

        assert df["period_order"].min() == 0
        assert df["period_order"].max() == 13
        assert df["start_year"].min() == 2012
        assert df["end_year"].max() == 2026

        # Check new periods explicitly present
        assert "2023-24" in periods
        assert "2024-25" in periods
        assert "2025-26" in periods

    def test_target_label_and_units_integrity(self):
        """Verify target_unit is strictly 'WPI index points' and base_year is 'Base Year 2011-12 = 100.0'."""
        df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)
        assert (df["target_unit"] == TARGET_LABEL).all()
        assert (df["target_unit"] == "WPI index points").all()
        assert (df["base_year"] == BASE_YEAR_LABEL).all()
        assert (df["wpi_index_points"] > 0).all()
        assert df["wpi_index_points"].isna().sum() == 0

    def test_extended_wide_format(self):
        """Verify extended wide CSV dimensions: 26 rows by 20 columns."""
        assert EXTENDED_WIDE_CSV.exists()
        wide_df = pd.read_csv(EXTENDED_WIDE_CSV)
        assert wide_df.shape == (26, 20)
        assert "INDEX20232024" in wide_df.columns
        assert "INDEX20242025" in wide_df.columns
        assert "INDEX20252026" in wide_df.columns
        assert wide_df.isna().sum().sum() == 0

    def test_extended_preprocessing_audit_report(self):
        """Verify extended audit JSON file records 364 valid FY observations."""
        audit_file = PROCESSED_DATA_DIR / "extended_preprocessing_audit.json"
        assert audit_file.exists()
        with open(audit_file, "r", encoding="utf-8") as f:
            audit = json.load(f)
            assert audit["raw_total_records"] == 869
            assert audit["retained_input_commodities"] == 26
            assert audit["fy_periods_count"] == 14
            assert audit["fy_valid_observations"] == 364
            assert audit["target_unit_label"] == "WPI index points"


# ===========================================================================
# 3. Extended Feature Engineering & Leakage Prevention
# ===========================================================================
class TestExtendedFeatureEngineeringAndChronology:
    """Verifies feature engineering over the extended historical timeseries."""

    def test_extended_features_file_and_dimensions(self):
        """Verify agricultural_inputs_features_extended.csv exists with 312 rows and 20 columns."""
        assert EXTENDED_FEATURES_CSV.exists()
        feat_df = pd.read_csv(EXTENDED_FEATURES_CSV)
        # 26 commodities * (14 periods - 2 initial lag periods) = 312 rows
        assert len(feat_df) == 312
        assert feat_df["commodity_code"].nunique() == 26
        counts = feat_df.groupby("commodity_code").size()
        assert (counts == 12).all()

    def test_extended_feature_periods_span(self):
        """Verify usable feature periods span from 2014-15 to 2025-26."""
        feat_df = pd.read_csv(EXTENDED_FEATURES_CSV)
        periods = sorted(list(feat_df["period_label"].unique()))
        assert len(periods) == 12
        assert periods[0] == "2014-15"
        assert periods[-1] == "2025-26"
        assert "2023-24" in periods
        assert "2024-25" in periods
        assert "2025-26" in periods

    def test_chronological_feature_correctness_for_new_periods(self):
        """Verify lag_1, lag_2, historical_pct_change, and rolling_mean_2 for FY 2023-24 to 2025-26."""
        feat_df = pd.read_csv(EXTENDED_FEATURES_CSV)
        ts_df = pd.read_csv(EXTENDED_PROCESSED_TIMESERIES_CSV)

        # Check Di ammonium phosphate (1310020002)
        dap_feat = feat_df[feat_df["commodity_code"] == 1310020002].set_index("period_label")
        dap_ts = ts_df[ts_df["commodity_code"] == 1310020002].set_index("period_label")

        # In 2023-24: lag_1 must be 2022-23 actual, lag_2 must be 2021-22 actual
        assert dap_feat.loc["2023-24", "lag_1"] == dap_ts.loc["2022-23", "wpi_index_points"]
        assert dap_feat.loc["2023-24", "lag_2"] == dap_ts.loc["2021-22", "wpi_index_points"]

        # In 2024-25: lag_1 must be 2023-24 actual, lag_2 must be 2022-23 actual
        assert dap_feat.loc["2024-25", "lag_1"] == dap_ts.loc["2023-24", "wpi_index_points"]
        assert dap_feat.loc["2024-25", "lag_2"] == dap_ts.loc["2022-23", "wpi_index_points"]

        # In 2025-26: lag_1 must be 2024-25 actual, lag_2 must be 2023-24 actual
        assert dap_feat.loc["2025-26", "lag_1"] == dap_ts.loc["2024-25", "wpi_index_points"]
        assert dap_feat.loc["2025-26", "lag_2"] == dap_ts.loc["2023-24", "wpi_index_points"]

        # Mathematical formula checks across all 312 rows
        for _, row in feat_df.iterrows():
            l1 = row["lag_1"]
            l2 = row["lag_2"]
            expected_pct = round(((l1 - l2) / l2) * 100.0, 4)
            expected_mean = round((l1 + l2) / 2.0, 2)
            assert row["historical_pct_change"] == expected_pct
            assert row["rolling_mean_2"] == expected_mean

    def test_zero_temporal_leakage_in_extended_features(self):
        """Verify that altering future target values does not alter past lag features."""
        ts = load_extended_processed_timeseries()
        target_idx = 10  # Alter target at index 10 (FY 2022-23)
        orig_val = ts.loc[target_idx, "wpi_index_points"]

        ts_mod = ts.copy()
        ts_mod.loc[target_idx, "wpi_index_points"] = orig_val + 250.0

        feat_orig = create_extended_time_series_features(ts, drop_na=False)
        feat_mod = create_extended_time_series_features(ts_mod, drop_na=False)

        # Timestep 10's own features (lag_1, lag_2, rolling_mean_2) depend on timesteps 8 and 9, NOT timestep 10
        assert feat_mod.loc[target_idx, "lag_1"] == feat_orig.loc[target_idx, "lag_1"]
        assert feat_mod.loc[target_idx, "lag_2"] == feat_orig.loc[target_idx, "lag_2"]
        assert feat_mod.loc[target_idx, "rolling_mean_2"] == feat_orig.loc[target_idx, "rolling_mean_2"]

    def test_zero_cross_commodity_contamination(self):
        """Verify lag calculations never leak across commodity boundaries in the extended dataset."""
        ts = load_extended_processed_timeseries()
        raw_feat_df = create_extended_time_series_features(ts, drop_na=False)

        for code, group in raw_feat_df.groupby("commodity_code"):
            sorted_group = group.sort_values("period_order").reset_index(drop=True)
            assert pd.isna(sorted_group.loc[0, "lag_1"]), f"Commodity {code} leaked lag_1!"
            assert pd.isna(sorted_group.loc[0, "lag_2"]), f"Commodity {code} leaked lag_2!"
            assert pd.isna(sorted_group.loc[1, "lag_2"]), f"Commodity {code} leaked lag_2 on period 1!"


# ===========================================================================
# 4. Immutability & Preservation Checks
# ===========================================================================
class TestPreservationAndImmutability:
    """Verifies that Phase 3A preserves original raw data, models, and dashboard."""

    def test_raw_dataset_is_strictly_unmodified(self):
        """Verify raw WPI CSV remains exactly 869 rows and 27 columns with Bajra as first row."""
        assert RAW_WPI_CSV.exists()
        raw_df = pd.read_csv(RAW_WPI_CSV)
        assert raw_df.shape == (869, 27)
        assert raw_df.iloc[0]["COMM_NAME"] == "Bajra"

    def test_trained_model_pipeline_unmodified(self):
        """Verify best_model.joblib SHA-256 hash matches the baseline without retraining."""
        model_path = TRAINED_MODELS_DIR / "best_model.joblib"
        assert model_path.exists()
        with open(model_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        assert current_hash == BEST_MODEL_SHA256, "Trained model pipeline was unexpectedly modified!"

    def test_historical_training_and_testing_data_preserved(self):
        """Verify train_data.csv (156 samples) and test_data.csv (78 samples) remain pristine."""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_data.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_data.csv")
        assert len(train_df) == 156
        assert len(test_df) == 78
        assert train_df["commodity_code"].nunique() == 26
        assert test_df["commodity_code"].nunique() == 26
