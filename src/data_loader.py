"""Data loader and inspection module for Indian Government Wholesale Price Index (WPI) data.

Safely ingests the raw DPIIT WPI CSV without altering original files,
identifies agricultural inputs vs. non-inputs, and extracts structured
time-series while maintaining explicit 'WPI index points' labeling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    CommodityMetadata,
    CY_COLUMNS,
    CY_LABELS,
    EXTENDED_FY_COLUMNS,
    EXTENDED_FY_LABELS,
    FY_COLUMNS,
    FY_LABELS,
    RAW_WPI_CSV,
    STAGED_WPI_XLS,
    TARGET_LABEL,
    UNPOPULATED_COLUMNS,
)


def load_raw_wpi_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw Indian Government Wholesale Price Index dataset.

    Args:
        filepath: Path to the CSV file. If None, uses RAW_WPI_CSV.

    Returns:
        pd.DataFrame: Raw dataset with string 'NA' values replaced by np.nan
        for all numerical index columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    path = Path(filepath) if filepath else RAW_WPI_CSV
    if not path.exists():
        raise FileNotFoundError(f"Raw WPI dataset not found at: {path}")

    df = pd.read_csv(path)

    # Validate essential schema
    required_cols = {"COMM_NAME", "COMM_CODE", "COMM_WT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required schema columns: {missing}")

    # Convert numeric index columns, treating string 'NA' as NaN
    index_cols = [c for c in df.columns if c.startswith("INDEX")]
    for col in index_cols:
        df[col] = pd.to_numeric(df[col].replace("NA", pd.NA), errors="coerce")

    # Ensure COMM_CODE is integer type
    df["COMM_CODE"] = pd.to_numeric(df["COMM_CODE"], errors="raise").astype("int64")
    df["COMM_WT"] = pd.to_numeric(df["COMM_WT"], errors="raise").astype("float64")

    return df


def is_agricultural_input(commodity_code: int) -> bool:
    """Check if a commodity code corresponds to a verified agricultural input."""
    return int(commodity_code) in AGRICULTURAL_INPUT_CATALOG


def get_agricultural_input_catalog() -> Dict[int, CommodityMetadata]:
    """Return the dictionary of verified agricultural input commodities."""
    return AGRICULTURAL_INPUT_CATALOG.copy()


def get_agricultural_inputs_df(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Filter the WPI dataset to include only verified agricultural input commodities.

    Appends metadata columns: input_category, suggested_unit, and description.

    Args:
        df: WPI DataFrame. If None, loads from RAW_WPI_CSV.

    Returns:
        pd.DataFrame: Filtered DataFrame containing agricultural inputs with metadata.
    """
    if df is None:
        df = load_raw_wpi_data()

    valid_codes = list(AGRICULTURAL_INPUT_CATALOG.keys())
    agri_df = df[df["COMM_CODE"].isin(valid_codes)].copy()

    # Add metadata
    agri_df["input_category"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].category
    )
    agri_df["suggested_unit"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].suggested_unit
    )
    agri_df["description"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].description
    )

    return agri_df


def extract_commodity_series(
    df: pd.DataFrame,
    commodity_code: int,
    series_type: str = "FY",
) -> pd.DataFrame:
    """Extract a chronological time series for a specific commodity.

    CRITICAL: Preserves original government data and source-period labels.
    Never mixes Financial Year (Apr-Mar) and Calendar Year (Jan-Dec) observations.
    Explicitly labels the target values as 'WPI index points' (NOT rupees).

    Args:
        df: WPI DataFrame loaded via load_raw_wpi_data.
        commodity_code: 10-digit DPIIT commodity code.
        series_type: Either 'FY' (Financial Year, 2012-13 to 2022-23)
                            or 'CY' (Calendar Year, 2013 to 2022).

    Returns:
        pd.DataFrame: Chronologically ordered DataFrame with columns:
            - 'period_key': raw column name (e.g., 'INDEX20122013')
            - 'period_label': human-readable period (e.g., '2012-13')
            - 'period_type': 'FY' or 'CY'
            - 'wpi_index_points': numeric index value (Base 2011-12 = 100.0)
            - 'commodity_name': official commodity name
            - 'commodity_code': integer code

    Raises:
        ValueError: If series_type is invalid or commodity_code is not found.
    """
    norm_type = series_type.strip().upper()
    if norm_type not in ("FY", "CY"):
        raise ValueError(
            f"Invalid series_type '{series_type}'. Must be either 'FY' (Financial Year) or 'CY' (Calendar Year)."
        )

    match = df[df["COMM_CODE"] == int(commodity_code)]
    if match.empty:
        raise ValueError(f"Commodity code {commodity_code} not found in WPI dataset.")

    row = match.iloc[0]
    comm_name = str(row["COMM_NAME"])

    columns = FY_COLUMNS if norm_type == "FY" else CY_COLUMNS
    labels = FY_LABELS if norm_type == "FY" else CY_LABELS

    records = []
    for col in columns:
        val = row[col]
        # Reject missing observations without fabricating data
        if pd.isna(val):
            continue
        records.append(
            {
                "period_key": col,
                "period_label": labels.get(col, col),
                "period_type": norm_type,
                "target_unit": TARGET_LABEL,  # Explicitly 'WPI index points'
                "wpi_index_points": float(val),
                "commodity_name": comm_name,
                "commodity_code": int(commodity_code),
            }
        )

    return pd.DataFrame(records)


def get_latest_observed_wpi(
    df: pd.DataFrame,
    commodity_code: int,
    series_type: str = "FY",
) -> Tuple[float, str]:
    """Retrieve the latest observed historical WPI value and period for a commodity.

    Args:
        df: WPI DataFrame.
        commodity_code: 10-digit commodity code.
        series_type: 'FY' or 'CY'.

    Returns:
        Tuple[float, str]: (latest_wpi_index_points, period_label).
    """
    series_df = extract_commodity_series(df, commodity_code, series_type=series_type)
    if series_df.empty:
        raise ValueError(
            f"No valid observations found for commodity {commodity_code} in {series_type} series."
        )
    latest_row = series_df.iloc[-1]
    return float(latest_row["wpi_index_points"]), str(latest_row["period_label"])


def load_staged_wpi_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the staged official DPIIT WPI dataset (FY 2012-13 to 2025-26).

    Args:
        filepath: Path to the staged XLS file. Defaults to STAGED_WPI_XLS.

    Returns:
        pd.DataFrame: Clean staged dataset with 869 commodities and all 14 FY columns.

    Raises:
        FileNotFoundError: If the staged XLS file does not exist.
        ValueError: If required columns, 26 agricultural inputs, or index values are invalid.
    """
    path = Path(filepath) if filepath else STAGED_WPI_XLS
    if not path.exists():
        raise FileNotFoundError(f"Staged WPI dataset not found at: {path}")

    df = pd.read_excel(path)

    required_cols = {"COMM_NAME", "COMM_CODE", "COMM_WT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Staged dataset missing required schema columns: {missing}")

    missing_fy = set(EXTENDED_FY_COLUMNS) - set(df.columns)
    if missing_fy:
        raise ValueError(f"Staged dataset missing extended FY columns: {missing_fy}")

    df["COMM_CODE"] = pd.to_numeric(df["COMM_CODE"], errors="raise").astype("int64")
    df["COMM_WT"] = pd.to_numeric(df["COMM_WT"], errors="raise").astype("float64")

    for col in EXTENDED_FY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate that all 26 catalog commodities are present with non-null values
    valid_codes = set(AGRICULTURAL_INPUT_CATALOG.keys())
    present_codes = set(df["COMM_CODE"])
    missing_agri = valid_codes - present_codes
    if missing_agri:
        raise ValueError(f"Staged dataset missing agricultural input commodities: {missing_agri}")

    agri_subset = df[df["COMM_CODE"].isin(valid_codes)]
    null_counts = int(agri_subset[EXTENDED_FY_COLUMNS].isna().sum().sum())
    if null_counts > 0:
        raise ValueError(f"Staged dataset contains {null_counts} null values in agricultural input commodities.")

    return df


def get_extended_agricultural_inputs_df(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Filter the staged WPI dataset to include only verified agricultural input commodities.

    Appends metadata columns: input_category, suggested_unit, and description.

    Args:
        df: Staged WPI DataFrame. If None, loads from STAGED_WPI_XLS.

    Returns:
        pd.DataFrame: Filtered DataFrame containing 26 agricultural inputs across 14 FY periods.
    """
    if df is None:
        df = load_staged_wpi_data()

    valid_codes = list(AGRICULTURAL_INPUT_CATALOG.keys())
    agri_df = df[df["COMM_CODE"].isin(valid_codes)].copy()

    agri_df["input_category"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].category
    )
    agri_df["suggested_unit"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].suggested_unit
    )
    agri_df["description"] = agri_df["COMM_CODE"].map(
        lambda c: AGRICULTURAL_INPUT_CATALOG[c].description
    )

    return agri_df
