"""Unit tests for Stage 4 Feature Engineering module (src/feature_engineering.py).

Verifies:
1. Correctness of lag calculations (lag_1, lag_2).
2. Correctness of historical percentage change calculation.
3. Absence of data leakage in rolling averages and lag features.
4. Prevention of cross-commodity boundary contamination.
5. Strict temporal separation between training and testing periods (no overlap, strictly chronological).
6. Preservation of 'WPI index points' target variable and units.
7. Correct extraction of scikit-learn compatible feature matrices.
8. File persistence and summary metadata generation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DATA_DIR, TARGET_LABEL
from src.feature_engineering import (
    CORE_FEATURE_COLUMNS,
    DEFAULT_TEST_PERIODS,
    DEFAULT_TRAIN_PERIODS,
    TARGET_COLUMN,
    chronological_train_test_split,
    create_time_series_features,
    get_feature_matrices,
    load_processed_timeseries,
    save_feature_engineered_datasets,
    FeatureEngineeringSummary,
)


class TestFeatureCalculationsAndLeakagePrevention:
    """Tests for feature engineering logic and leakage prevention."""

    def test_lag_values_match_historical_series(self):
        """Verify lag_1 and lag_2 accurately reflect t-1 and t-2 observations."""
        ts = load_processed_timeseries()
        feat_df = create_time_series_features(ts, drop_na=False)

        # Check Di ammonium phosphate (1310020002)
        dap = feat_df[feat_df["commodity_code"] == 1310020002].sort_values("period_order").reset_index(drop=True)
        # Period 0: 2012-13 (raw WPI: 133.7) -> lags should be NaN
        assert pd.isna(dap.loc[0, "lag_1"])
        assert pd.isna(dap.loc[0, "lag_2"])

        # Period 1: 2013-14 (raw WPI: 132.4) -> lag_1 = 133.7, lag_2 = NaN
        assert dap.loc[1, "lag_1"] == 133.7
        assert pd.isna(dap.loc[1, "lag_2"])

        # Period 2: 2014-15 (raw WPI: 135.1) -> lag_1 = 132.4, lag_2 = 133.7
        assert dap.loc[2, "lag_1"] == 132.4
        assert dap.loc[2, "lag_2"] == 133.7

        # Period 10: 2022-23 (raw WPI: 164.0)
        # 2021-22 was 145.3, 2020-21 was 136.0
        assert dap.loc[10, "lag_1"] == 145.3
        assert dap.loc[10, "lag_2"] == 136.0

    def test_historical_percentage_change_calculation(self):
        """Verify historical percentage change formula: ((lag_1 - lag_2) / lag_2) * 100."""
        ts = load_processed_timeseries()
        feat_df = create_time_series_features(ts, drop_na=True)

        for _, row in feat_df.iterrows():
            l1 = row["lag_1"]
            l2 = row["lag_2"]
            expected_pct = round(((l1 - l2) / l2) * 100.0, 4)
            assert row["historical_pct_change"] == expected_pct

    def test_rolling_mean_excludes_current_target(self):
        """Verify rolling_mean_2 uses strictly past lags and does not depend on target y_t."""
        ts = load_processed_timeseries()
        feat_df = create_time_series_features(ts, drop_na=True)

        for _, row in feat_df.iterrows():
            l1 = row["lag_1"]
            l2 = row["lag_2"]
            expected_mean = round((l1 + l2) / 2.0, 2)
            assert row["rolling_mean_2"] == expected_mean

        # Invariance test: altering target y_t in input data does not change lag_1, lag_2, or rolling_mean_2
        ts_modified = ts.copy()
        target_idx = 5  # Modify target at index 5
        original_target = ts_modified.loc[target_idx, "wpi_index_points"]
        ts_modified.loc[target_idx, "wpi_index_points"] = original_target + 500.0  # Big change

        feat_modified = create_time_series_features(ts_modified, drop_na=False)
        # The features for row 5 must remain completely identical because features for row 5 use rows 3 and 4
        assert feat_modified.loc[target_idx, "lag_1"] == feat_df.loc[target_idx - 2, "lag_1"]
        assert feat_modified.loc[target_idx, "rolling_mean_2"] == feat_df.loc[target_idx - 2, "rolling_mean_2"]

    def test_zero_cross_commodity_contamination(self):
        """Verify lag calculations never leak across commodity boundaries."""
        ts = load_processed_timeseries()
        # Evaluate without dropping initial lag rows
        raw_feat_df = create_time_series_features(ts, drop_na=False)

        for code, group in raw_feat_df.groupby("commodity_code"):
            sorted_group = group.sort_values("period_order").reset_index(drop=True)
            # The first observation (period_order=0) of EVERY commodity must have NaN for lag_1 and lag_2
            assert pd.isna(sorted_group.loc[0, "lag_1"]), f"Commodity {code} leaked lag_1 from previous commodity!"
            assert pd.isna(sorted_group.loc[0, "lag_2"]), f"Commodity {code} leaked lag_2 from previous commodity!"
            # The second observation (period_order=1) must have NaN for lag_2
            assert pd.isna(sorted_group.loc[1, "lag_2"]), f"Commodity {code} leaked lag_2 on period 1!"

    def test_drop_na_row_reduction(self):
        """Verify dropping initial lag periods removes exactly 2 periods per commodity (26 * 2 = 52)."""
        ts = load_processed_timeseries()
        raw_feat_df = create_time_series_features(ts, drop_na=False)
        clean_feat_df = create_time_series_features(ts, drop_na=True)

        assert len(raw_feat_df) == 286
        assert len(clean_feat_df) == 286 - (26 * 2)  # 234 rows
        assert len(clean_feat_df) == 234

        # Every commodity must have exactly 9 remaining rows
        counts = clean_feat_df.groupby("commodity_code").size()
        assert (counts == 9).all()


class TestChronologicalTrainTestSplit:
    """Tests for chronological_train_test_split() verifying strict temporal isolation."""

    def test_default_split_dimensions_and_periods(self):
        """Verify train (156 samples, 6 years) and test (78 samples, 3 years) partitions."""
        feat_df = create_time_series_features(drop_na=True)
        train_df, test_df = chronological_train_test_split(feat_df)

        assert len(train_df) == 26 * 6  # 156 samples
        assert len(test_df) == 26 * 3   # 78 samples
        assert len(train_df) + len(test_df) == len(feat_df)

        # Period assertions
        assert list(train_df["period_label"].unique()) == DEFAULT_TRAIN_PERIODS
        assert list(test_df["period_label"].unique()) == DEFAULT_TEST_PERIODS

    def test_zero_period_overlap_and_strict_order(self):
        """Verify no overlap between train and test periods and strict chronological boundary."""
        feat_df = create_time_series_features(drop_na=True)
        train_df, test_df = chronological_train_test_split(feat_df)

        train_periods = set(train_df["period_label"].unique())
        test_periods = set(test_df["period_label"].unique())

        # Zero set intersection
        assert len(train_periods.intersection(test_periods)) == 0

        # Max train period order < min test period order
        assert train_df["period_order"].max() < test_df["period_order"].min()
        assert train_df["period_order"].max() == 7   # 2019-20
        assert test_df["period_order"].min() == 8    # 2020-21

        # Max train start_year < min test start_year
        assert train_df["start_year"].max() < test_df["start_year"].min()
        assert train_df["start_year"].max() == 2019
        assert test_df["start_year"].min() == 2020

    def test_single_commodity_split(self):
        """Verify chronological split for an individual commodity."""
        feat_df = create_time_series_features(drop_na=True)
        # HSD (1202000005)
        train_hsd, test_hsd = chronological_train_test_split(feat_df, commodity_code=1202000005)

        assert len(train_hsd) == 6
        assert len(test_hsd) == 3
        assert (train_hsd["commodity_code"] == 1202000005).all()
        assert (test_hsd["commodity_code"] == 1202000005).all()

    def test_reject_overlapping_periods(self):
        """Reject custom period lists with overlapping years."""
        feat_df = create_time_series_features(drop_na=True)
        with pytest.raises(ValueError, match="Overlapping periods found"):
            chronological_train_test_split(
                feat_df,
                train_periods=["2014-15", "2015-16", "2016-17"],
                test_periods=["2016-17", "2017-18"],
            )

    def test_reject_inverted_chronological_periods(self):
        """Reject split where training periods occur after testing periods."""
        feat_df = create_time_series_features(drop_na=True)
        with pytest.raises(ValueError, match="Chronological boundary violation"):
            chronological_train_test_split(
                feat_df,
                train_periods=["2021-22", "2022-23"],
                test_periods=["2014-15", "2015-16"],
            )

    def test_reject_nonexistent_commodity(self):
        """Reject single-commodity split when commodity code is invalid."""
        feat_df = create_time_series_features(drop_na=True)
        with pytest.raises(ValueError, match="not found in feature dataset"):
            chronological_train_test_split(feat_df, commodity_code=9999999999)


class TestTargetVariableIntegrityAndMatrixExtraction:
    """Tests verifying target units and scikit-learn matrix extraction."""

    def test_target_variable_unit_and_values(self):
        """Verify target remains strictly 'WPI index points' and non-empty."""
        feat_df = create_time_series_features(drop_na=True)
        train_df, test_df = chronological_train_test_split(feat_df)

        for partition in [train_df, test_df]:
            assert (partition["target_unit"] == TARGET_LABEL).all()
            assert (partition["target_unit"] == "WPI index points").all()
            assert TARGET_COLUMN in partition.columns
            assert (partition[TARGET_COLUMN] > 0).all()
            assert not partition[TARGET_COLUMN].isna().any()

    def test_get_feature_matrices(self):
        """Verify X_train, y_train, X_test, y_test shapes and columns."""
        feat_df = create_time_series_features(drop_na=True)
        train_df, test_df = chronological_train_test_split(feat_df)

        X_tr, y_tr, X_te, y_te = get_feature_matrices(train_df, test_df)

        assert X_tr.shape == (156, len(CORE_FEATURE_COLUMNS))
        assert y_tr.shape == (156,)
        assert X_te.shape == (78, len(CORE_FEATURE_COLUMNS))
        assert y_te.shape == (78,)

        assert list(X_tr.columns) == CORE_FEATURE_COLUMNS
        assert list(X_te.columns) == CORE_FEATURE_COLUMNS
        assert y_tr.name == TARGET_COLUMN
        assert y_te.name == TARGET_COLUMN

        # No nulls in matrices
        assert X_tr.isna().sum().sum() == 0
        assert X_te.isna().sum().sum() == 0
        assert y_tr.isna().sum() == 0
        assert y_te.isna().sum() == 0


class TestPersistenceAndArtifacts:
    """Tests for save_feature_engineered_datasets() output verification."""

    def test_feature_artifacts_saved_and_valid(self):
        """Verify that all 4 feature engineering artifacts exist and are readable."""
        feat_p, tr_p, te_p, sum_p = save_feature_engineered_datasets()

        for path in [feat_p, tr_p, te_p, sum_p]:
            assert path.exists()
            assert path.stat().st_size > 0

        # Check summary JSON contents
        with open(sum_p, "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["total_commodities"] == 26
            assert summary["raw_timeseries_rows"] == 286
            assert summary["dropped_initial_lag_rows"] == 52
            assert summary["usable_rows"] == 234
            assert summary["usable_observations_per_commodity"] == 9
            assert summary["train_samples_count"] == 156
            assert summary["test_samples_count"] == 78
            assert summary["target_unit_label"] == "WPI index points"
