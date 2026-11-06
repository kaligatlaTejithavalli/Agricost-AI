"""Unit tests for the data loader and catalog inspection module (src/data_loader.py).

Verifies:
1. Ingestion of the raw government dataset without data corruption.
2. Unaltered preservation of raw CSV observations.
3. Proper identification of unpopulated historical columns (INDEX2011, 2011-12, 2012).
4. Accurate filtering of agricultural input commodities vs. crop harvest outputs.
5. Strict time series extraction preserving 'WPI index points' and period labels.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    RAW_WPI_CSV,
    TARGET_LABEL,
)
from src.data_loader import (
    extract_commodity_series,
    get_agricultural_inputs_df,
    get_latest_observed_wpi,
    is_agricultural_input,
    load_raw_wpi_data,
)


class TestRawDatasetIngestion:
    """Tests for raw CSV loading and schema validation."""

    def test_raw_file_exists_and_loads(self):
        """Verify the raw CSV exists and loads with exact dimensions."""
        assert RAW_WPI_CSV.exists()
        df = load_raw_wpi_data(RAW_WPI_CSV)
        assert df.shape == (869, 27)
        assert "COMM_NAME" in df.columns
        assert "COMM_CODE" in df.columns
        assert "COMM_WT" in df.columns

    def test_unpopulated_columns_are_strictly_nan(self):
        """Verify that INDEX2011, INDEX20112012, and INDEX2012 are strictly NaN (not fabricated)."""
        df = load_raw_wpi_data()
        assert df["INDEX2011"].isna().all()
        assert df["INDEX20112012"].isna().all()
        assert df["INDEX2012"].isna().all()

    def test_populated_index_columns_have_no_missing_values(self):
        """Verify all 21 post-2012 index columns have complete historical observations."""
        df = load_raw_wpi_data()
        populated_cols = [
            "INDEX20122013",
            "INDEX2013",
            "INDEX20132014",
            "INDEX2014",
            "INDEX20142015",
            "INDEX2015",
            "INDEX20152016",
            "INDEX2016",
            "INDEX20162017",
            "INDEX2017",
            "INDEX20172018",
            "INDEX2018",
            "INDEX20182019",
            "INDEX2019",
            "INDEX20192020",
            "INDEX2020",
            "INDEX20202021",
            "INDEX2021",
            "INDEX20212022",
            "INDEX2022",
            "INDEX20222023",
        ]
        for col in populated_cols:
            assert df[col].notna().all(), f"Found unexpected missing values in {col}"
            assert (df[col] > 0).all(), f"Found non-positive index values in {col}"


class TestAgriculturalInputCatalog:
    """Tests ensuring clear differentiation between agricultural inputs and farm outputs / industrial goods."""

    def test_verified_inputs_are_recognized(self):
        """Verify key agricultural inputs are recognized in the catalog."""
        # Urea
        assert is_agricultural_input(1310020001) is True
        # DAP
        assert is_agricultural_input(1310020002) is True
        # High Speed Diesel (HSD)
        assert is_agricultural_input(1202000005) is True
        # Agricultural tractors
        assert is_agricultural_input(1318090001) is True
        # Insecticide & Pesticide
        assert is_agricultural_input(1310040001) is True
        # Electricity
        assert is_agricultural_input(1203000001) is True
        # Fodder
        assert is_agricultural_input(1102030008) is True

    def test_crop_outputs_are_excluded(self):
        """Verify that crop harvests / farm outputs are NOT categorized as inputs."""
        # Bajra (Food Grain)
        assert is_agricultural_input(1101010104) is False
        # Masur (Pulse)
        assert is_agricultural_input(1101010204) is False
        # Tomato (Vegetable)
        assert is_agricultural_input(1101020107) is False
        # Apple (Fruit)
        assert is_agricultural_input(1101020203) is False
        # Raw Jute (Fibre crop harvest)
        assert is_agricultural_input(1102010002) is False

    def test_unrelated_manufactured_goods_are_excluded(self):
        """Verify unrelated industrial products are excluded."""
        # Copper Concentrate
        assert is_agricultural_input(1103010006) is False
        # Toilet Soap
        assert is_agricultural_input(1310060001) is False

    def test_agricultural_inputs_df_filter(self):
        """Verify get_agricultural_inputs_df returns only items in the catalog."""
        agri_df = get_agricultural_inputs_df()
        assert len(agri_df) == len(AGRICULTURAL_INPUT_CATALOG)
        assert set(agri_df["COMM_CODE"]) == set(AGRICULTURAL_INPUT_CATALOG.keys())
        assert "input_category" in agri_df.columns
        assert "suggested_unit" in agri_df.columns


class TestCommoditySeriesExtraction:
    """Tests for chronological time-series extraction from the dataset."""

    def test_financial_year_series_extraction(self):
        """Verify extracting Financial Year (FY) series."""
        df = load_raw_wpi_data()
        # Di ammonium phosphate (1310020002)
        dap_series = extract_commodity_series(df, 1310020002, series_type="FY")

        # Must have exactly 11 financial year points (2012-13 to 2022-23)
        assert len(dap_series) == 11
        assert dap_series.iloc[0]["period_label"] == "2012-13"
        assert dap_series.iloc[-1]["period_label"] == "2022-23"
        assert (dap_series["period_type"] == "FY").all()

        # Explicit target unit assertion (Task 2)
        assert (dap_series["target_unit"] == TARGET_LABEL).all()
        assert (dap_series["target_unit"] == "WPI index points").all()

        # Verify exact historical values from raw file
        assert dap_series.iloc[0]["wpi_index_points"] == 133.7
        assert dap_series.iloc[-1]["wpi_index_points"] == 164.0

    def test_calendar_year_series_extraction(self):
        """Verify extracting Calendar Year (CY) series."""
        df = load_raw_wpi_data()
        # HSD (1202000005)
        hsd_series = extract_commodity_series(df, 1202000005, series_type="CY")

        # Must have exactly 10 calendar year points (2013 to 2022)
        assert len(hsd_series) == 10
        assert hsd_series.iloc[0]["period_label"] == "2013"
        assert hsd_series.iloc[-1]["period_label"] == "2022"
        assert (hsd_series["period_type"] == "CY").all()
        assert (hsd_series["target_unit"] == "WPI index points").all()

    def test_reject_invalid_series_type(self):
        """Reject invalid series type parameters."""
        df = load_raw_wpi_data()
        with pytest.raises(ValueError, match="Invalid series_type"):
            extract_commodity_series(df, 1310020001, series_type="QUARTERLY")

    def test_reject_nonexistent_commodity_code(self):
        """Reject non-existent commodity codes."""
        df = load_raw_wpi_data()
        with pytest.raises(ValueError, match="not found in WPI dataset"):
            extract_commodity_series(df, 9999999999, series_type="FY")

    def test_latest_observed_wpi(self):
        """Verify retrieval of latest observed historical index points."""
        df = load_raw_wpi_data()
        val, period = get_latest_observed_wpi(df, 1310020001, series_type="FY")
        # Urea 2022-23 index point
        assert period == "2022-23"
        assert val == 115.1
