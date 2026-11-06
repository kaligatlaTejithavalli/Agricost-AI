"""Unit tests for Stage 3 Data Preprocessing module (src/preprocessor.py).

Verifies:
1. Raw dataset audit metrics (total records, zero duplicates, missing counts).
2. Complete exclusion of non-input commodities (843 excluded, 26 retained).
3. Longitudinal time-series reshaping into tidy format.
4. Strict chronological ordering of historical periods.
5. Uncompromised preservation of raw WPI index values and 'WPI index points' target label.
6. Clean artifact generation under data/processed/ without mutating raw data.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    FY_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_WPI_CSV,
    TARGET_LABEL,
    UNPOPULATED_COLUMNS,
)
from src.data_loader import load_raw_wpi_data
from src.preprocessor import (
    audit_raw_data,
    preprocess_wpi_timeseries,
    preprocess_wpi_wide,
    save_processed_data,
    PreprocessingAudit,
)


class TestRawDataAudit:
    """Tests for audit_raw_data() verifying dataset health and provenance."""

    def test_audit_records_and_duplicates(self):
        """Verify record counts, zero duplicate commodity codes/names, and unpopulated counts."""
        audit = audit_raw_data()
        assert isinstance(audit, PreprocessingAudit)
        assert audit.raw_total_records == 869
        assert audit.raw_columns_count == 27
        assert audit.duplicate_commodity_codes == 0
        assert audit.duplicate_commodity_names == 0

        # Exactly 3 unpopulated columns: INDEX2011, INDEX20112012, INDEX2012 (869 * 3 = 2607)
        assert audit.unpopulated_columns == list(UNPOPULATED_COLUMNS)
        assert audit.unpopulated_missing_count == 869 * 3

    def test_audit_agricultural_filtering(self):
        """Verify exactly 26 agricultural input commodities are retained and 843 excluded."""
        audit = audit_raw_data()
        assert audit.retained_input_commodities == 26
        assert audit.excluded_commodities_count == 843
        assert audit.retained_input_commodities + audit.excluded_commodities_count == 869

        # Verify category distribution
        cats = audit.categories_breakdown
        assert cats["Fertilizer"] == 7
        assert cats["Farm Machinery"] == 6
        assert cats["Livestock Input"] == 4
        assert cats["Fuel & Power"] == 3
        assert cats["Pesticide / Agrochemical"] == 3
        assert cats["Fertilizer (Group Index)"] == 1
        assert cats["Farm Machinery (Group Index)"] == 1
        assert cats["Pesticide / Agrochemical (Group Index)"] == 1

    def test_audit_observations_count(self):
        """Verify total observation count for FY (26 * 11 = 286) and CY (26 * 10 = 260)."""
        audit = audit_raw_data()
        assert audit.fy_periods_count == 11
        assert audit.fy_valid_observations == 26 * 11
        assert audit.cy_periods_count == 10
        assert audit.cy_valid_observations == 26 * 10
        assert audit.target_unit_label == "WPI index points"


class TestTimeseriesReshapingAndOrdering:
    """Tests for preprocess_wpi_timeseries() verifying longitudinal structure and chronology."""

    def test_fy_timeseries_structure_and_types(self):
        """Verify Financial Year timeseries rows, columns, and datatypes."""
        ts = preprocess_wpi_timeseries(series_type="FY")

        # 26 commodities * 11 FY periods = 286 rows
        assert len(ts) == 286
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
        assert list(ts.columns) == expected_cols

        # No missing values in any column
        assert ts.isna().sum().sum() == 0

        # Target label check (Task 2 & Important constraint)
        assert (ts["target_unit"] == TARGET_LABEL).all()
        assert (ts["target_unit"] == "WPI index points").all()

    def test_strict_chronological_ordering(self):
        """Verify that every commodity's observations are strictly ordered by period_order and year."""
        ts = preprocess_wpi_timeseries(series_type="FY")

        for code, group in ts.groupby("commodity_code"):
            # Must have exactly 11 observations
            assert len(group) == 11

            orders = group["period_order"].tolist()
            assert orders == list(range(11)), f"Period orders not sequential for {code}"

            start_years = group["start_year"].tolist()
            assert start_years == list(range(2012, 2023)), f"Start years not strictly monotonic for {code}"

            labels = group["period_label"].tolist()
            expected_labels = [
                "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
                "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"
            ]
            assert labels == expected_labels

    def test_original_wpi_values_preserved(self):
        """Verify that index points match raw government values without modification."""
        raw_df = load_raw_wpi_data()
        ts = preprocess_wpi_timeseries(raw_df, series_type="FY")

        # Check Urea (1310020001) in 2022-23 (raw was 115.1)
        urea_2223 = ts[(ts["commodity_code"] == 1310020001) & (ts["period_label"] == "2022-23")]
        assert len(urea_2223) == 1
        assert urea_2223.iloc[0]["wpi_index_points"] == 115.1

        # Check DAP (1310020002) in 2022-23 (raw was 164.0)
        dap_2223 = ts[(ts["commodity_code"] == 1310020002) & (ts["period_label"] == "2022-23")]
        assert len(dap_2223) == 1
        assert dap_2223.iloc[0]["wpi_index_points"] == 164.0

        # Check HSD (1202000005) in 2022-23 (raw was 191.4)
        hsd_2223 = ts[(ts["commodity_code"] == 1202000005) & (ts["period_label"] == "2022-23")]
        assert len(hsd_2223) == 1
        assert hsd_2223.iloc[0]["wpi_index_points"] == 191.4

    def test_cy_timeseries_structure(self):
        """Verify Calendar Year timeseries (10 periods: 2013 to 2022)."""
        ts_cy = preprocess_wpi_timeseries(series_type="CY")
        assert len(ts_cy) == 260
        assert (ts_cy["series_type"] == "CY").all()

        sample_comm = ts_cy[ts_cy["commodity_code"] == 1202000005]  # HSD
        assert len(sample_comm) == 10
        assert sample_comm.iloc[0]["period_label"] == "2013"
        assert sample_comm.iloc[-1]["period_label"] == "2022"

    def test_reject_invalid_series_type(self):
        """Reject unsupported series types."""
        with pytest.raises(ValueError, match="Invalid series_type"):
            preprocess_wpi_timeseries(series_type="MONTHLY")


class TestWideFormatPreprocessing:
    """Tests for preprocess_wpi_wide()."""

    def test_wide_format_dimensions_and_metadata(self):
        """Verify wide format dimensions (26 rows, 17 columns for FY)."""
        wide = preprocess_wpi_wide(series_type="FY")
        assert wide.shape == (26, 17)
        assert "commodity_code" in wide.columns
        assert "commodity_name" in wide.columns
        assert "input_category" in wide.columns
        assert "INDEX20122013" in wide.columns
        assert "INDEX20222023" in wide.columns
        assert wide.isna().sum().sum() == 0


class TestProcessedDataPersistence:
    """Tests for save_processed_data() and filesystem integrity."""

    def test_processed_files_exist_and_readable(self):
        """Verify that all 4 artifacts are generated and non-empty in data/processed/."""
        fy_p, cy_p, wide_p, audit_p = save_processed_data()

        for path in [fy_p, cy_p, wide_p, audit_p]:
            assert path.exists(), f"Expected artifact {path} does not exist"
            assert path.stat().st_size > 0, f"Artifact {path} is empty"

        # Check audit json validity
        with open(audit_p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["raw_total_records"] == 869
            assert data["retained_input_commodities"] == 26
            assert data["excluded_commodities_count"] == 843

    def test_raw_dataset_is_strictly_unmodified(self):
        """Verify raw CSV file byte size and record count remain pristine."""
        raw_df = pd.read_csv(RAW_WPI_CSV)
        assert raw_df.shape == (869, 27)
        # First row Bajra check
        assert raw_df.iloc[0]["COMM_NAME"] == "Bajra"
