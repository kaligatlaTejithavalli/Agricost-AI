"""Data Preprocessing Pipeline for AgriCost-AI.

Stage 3 Preprocessing Module:
1. Ingests raw Wholesale Price Index (WPI) dataset without mutating the original source.
2. Audits dataset for duplicates, missing values, non-numeric values, and schema integrity.
3. Isolates verified agricultural input commodities using the central catalog.
4. Segregates and structures chronological time series (Financial Years 2012-13 to 2022-23
   and Calendar Years 2013 to 2022) without inventing or fabricating missing observations.
5. Emits normalized, leak-free, long-format datasets and audit summaries under data/processed/.
6. Enforces that all index target values remain strictly labeled as 'WPI index points', NEVER rupees.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    CY_COLUMNS,
    CY_LABELS,
    EXTENDED_FY_COLUMNS,
    EXTENDED_FY_LABELS,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_WIDE_CSV,
    FY_COLUMNS,
    FY_LABELS,
    PROCESSED_DATA_DIR,
    RAW_WPI_CSV,
    STAGED_WPI_XLS,
    TARGET_LABEL,
    UNPOPULATED_COLUMNS,
)
from src.data_loader import (
    get_agricultural_inputs_df,
    get_extended_agricultural_inputs_df,
    load_raw_wpi_data,
    load_staged_wpi_data,
)


@dataclass(frozen=True)
class PreprocessingAudit:
    """Encapsulates the statistical and schema audit results for the preprocessing pipeline."""

    raw_total_records: int
    raw_columns_count: int
    duplicate_commodity_codes: int
    duplicate_commodity_names: int
    unpopulated_columns: List[str]
    unpopulated_missing_count: int
    retained_input_commodities: int
    excluded_commodities_count: int
    categories_breakdown: Dict[str, int]
    fy_periods_count: int
    fy_valid_observations: int
    cy_periods_count: int
    cy_valid_observations: int
    min_wpi_value_fy: float
    max_wpi_value_fy: float
    mean_wpi_value_fy: float
    target_unit_label: str
    dataset_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_raw_data(df: Optional[pd.DataFrame] = None) -> PreprocessingAudit:
    """Perform a rigorous schema, nullness, duplicate, and distribution audit of the WPI data.

    Args:
        df: Raw or loaded WPI DataFrame. If None, loaded via load_raw_wpi_data().

    Returns:
        PreprocessingAudit: Audit summary metrics.
    """
    if df is None:
        df = load_raw_wpi_data()

    raw_total = len(df)
    dup_codes = int(df["COMM_CODE"].duplicated().sum())
    dup_names = int(df["COMM_NAME"].duplicated().sum())

    # Audit unpopulated columns
    unpop_count = 0
    for col in UNPOPULATED_COLUMNS:
        if col in df.columns:
            unpop_count += int(df[col].isna().sum())

    # Retained agricultural inputs
    agri_df = get_agricultural_inputs_df(df)
    retained_count = len(agri_df)
    excluded_count = raw_total - retained_count

    categories_breakdown = agri_df["input_category"].value_counts().to_dict()

    # Calculate observations and numerical ranges for FY
    fy_obs = int(agri_df[FY_COLUMNS].notna().sum().sum())
    cy_obs = int(agri_df[CY_COLUMNS].notna().sum().sum())

    all_fy_values = agri_df[FY_COLUMNS].values.flatten()
    clean_fy = all_fy_values[~np.isnan(all_fy_values)]

    min_val = float(np.min(clean_fy)) if len(clean_fy) > 0 else 0.0
    max_val = float(np.max(clean_fy)) if len(clean_fy) > 0 else 0.0
    mean_val = float(np.mean(clean_fy)) if len(clean_fy) > 0 else 0.0

    return PreprocessingAudit(
        raw_total_records=raw_total,
        raw_columns_count=len(df.columns),
        duplicate_commodity_codes=dup_codes,
        duplicate_commodity_names=dup_names,
        unpopulated_columns=list(UNPOPULATED_COLUMNS),
        unpopulated_missing_count=unpop_count,
        retained_input_commodities=retained_count,
        excluded_commodities_count=excluded_count,
        categories_breakdown=categories_breakdown,
        fy_periods_count=len(FY_COLUMNS),
        fy_valid_observations=fy_obs,
        cy_periods_count=len(CY_COLUMNS),
        cy_valid_observations=cy_obs,
        min_wpi_value_fy=round(min_val, 2),
        max_wpi_value_fy=round(max_val, 2),
        mean_wpi_value_fy=round(mean_val, 2),
        target_unit_label=TARGET_LABEL,
        dataset_source="Office of the Economic Adviser, DPIIT, Ministry of Commerce & Industry, Base 2011-12=100",
    )


def preprocess_wpi_timeseries(
    df: Optional[pd.DataFrame] = None,
    series_type: str = "FY",
) -> pd.DataFrame:
    """Extract, filter, and reshape agricultural input data into a clean, chronologically sorted time-series.

    Converts wide-format historical columns into a tidy longitudinal representation
    where each row represents one observation for a specific commodity and historical period.

    CRITICAL RULES:
    1. Preserves original WPI values strictly as 'WPI index points' (NEVER converts to rupees).
    2. Preserves commodity codes and categories.
    3. Rejects missing observation fabrication: only strictly observed values are kept.
    4. Enforces chronological order of periods.

    Args:
        df: WPI DataFrame. If None, loaded via load_raw_wpi_data().
        series_type: 'FY' (Financial Year, 2012-13 to 2022-23) or 'CY' (Calendar Year, 2013 to 2022).

    Returns:
        pd.DataFrame: Long-format time series dataset with columns:
            - commodity_code (int64)
            - commodity_name (str)
            - commodity_weight (float64)
            - input_category (str)
            - suggested_unit (str)
            - description (str)
            - series_type (str)
            - period_key (str)
            - period_label (str)
            - period_order (int)
            - start_year (int)
            - end_year (int)
            - wpi_index_points (float64)
            - target_unit (str)
            - base_year (str)
            - data_source (str)

    Raises:
        ValueError: If series_type is invalid.
    """
    norm_type = series_type.strip().upper()
    if norm_type not in ("FY", "CY"):
        raise ValueError(f"Invalid series_type '{series_type}'. Must be 'FY' or 'CY'.")

    if df is None:
        df = load_raw_wpi_data()

    agri_df = get_agricultural_inputs_df(df)

    columns = FY_COLUMNS if norm_type == "FY" else CY_COLUMNS
    labels = FY_LABELS if norm_type == "FY" else CY_LABELS

    rows = []
    for _, item in agri_df.iterrows():
        code = int(item["COMM_CODE"])
        name = str(item["COMM_NAME"])
        weight = float(item["COMM_WT"])
        category = str(item["input_category"])
        suggested_unit = str(item["suggested_unit"])
        description = str(item["description"])

        for order, col in enumerate(columns):
            val = item[col]
            if pd.isna(val):
                # Never fabricate or forward-fill missing values
                continue

            float_val = float(val)
            if float_val <= 0:
                raise ValueError(
                    f"Invalid non-positive WPI value {float_val} for {name} ({code}) in {col}."
                )

            lbl = labels[col]
            if norm_type == "FY":
                # Parse '2012-13' -> start_year=2012, end_year=2013
                s_yr = int(lbl.split("-")[0])
                e_yr = int("20" + lbl.split("-")[1])
            else:
                # Parse '2013' -> start_year=2013, end_year=2013
                s_yr = int(lbl)
                e_yr = int(lbl)

            rows.append(
                {
                    "commodity_code": code,
                    "commodity_name": name,
                    "commodity_weight": weight,
                    "input_category": category,
                    "suggested_unit": suggested_unit,
                    "description": description,
                    "series_type": norm_type,
                    "period_key": col,
                    "period_label": lbl,
                    "period_order": order,
                    "start_year": s_yr,
                    "end_year": e_yr,
                    "wpi_index_points": round(float_val, 2),
                    "target_unit": TARGET_LABEL,
                    "base_year": BASE_YEAR_LABEL,
                    "data_source": "DPIIT, Ministry of Commerce & Industry, Govt of India",
                }
            )

    processed_df = pd.DataFrame(rows)

    # Sort strictly chronologically by commodity_code and period_order
    processed_df.sort_values(
        by=["commodity_code", "period_order"],
        ascending=[True, True],
        inplace=True,
    )
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df


def preprocess_wpi_wide(
    df: Optional[pd.DataFrame] = None,
    series_type: str = "FY",
) -> pd.DataFrame:
    """Create a clean wide-format DataFrame for agricultural inputs with explicit metadata.

    Args:
        df: WPI DataFrame.
        series_type: 'FY' or 'CY'.

    Returns:
        pd.DataFrame: Clean wide format containing commodity metadata and chronological periods.
    """
    norm_type = series_type.strip().upper()
    if df is None:
        df = load_raw_wpi_data()

    agri_df = get_agricultural_inputs_df(df)
    columns = FY_COLUMNS if norm_type == "FY" else CY_COLUMNS

    meta_cols = [
        "COMM_CODE",
        "COMM_NAME",
        "COMM_WT",
        "input_category",
        "suggested_unit",
        "description",
    ]
    wide_df = agri_df[meta_cols + columns].copy()

    # Rename metadata columns for uniformity
    rename_map = {
        "COMM_CODE": "commodity_code",
        "COMM_NAME": "commodity_name",
        "COMM_WT": "commodity_weight",
    }
    wide_df.rename(columns=rename_map, inplace=True)
    wide_df.sort_values(by="commodity_code", inplace=True)
    wide_df.reset_index(drop=True, inplace=True)

    return wide_df


def save_processed_data(
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Path]:
    """Execute complete preprocessing and save artifacts to data/processed/.

    Saves:
        1. agricultural_inputs_wpi_fy_timeseries.csv (Financial Year long format)
        2. agricultural_inputs_wpi_cy_timeseries.csv (Calendar Year long format)
        3. agricultural_inputs_wpi_fy_wide.csv (Financial Year wide format)
        4. preprocessing_audit.json (Detailed audit and verification report)

    Args:
        output_dir: Directory where processed files will be written. Defaults to PROCESSED_DATA_DIR.

    Returns:
        Tuple of paths to the 4 created artifact files.
    """
    target_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_wpi_data()

    # 1. Generate Audit
    audit = audit_raw_data(raw_df)
    audit_file = target_dir / "preprocessing_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, indent=2)

    # 2. Generate Long Format (FY Series)
    fy_timeseries = preprocess_wpi_timeseries(raw_df, series_type="FY")
    fy_file = target_dir / "agricultural_inputs_wpi_fy_timeseries.csv"
    fy_timeseries.to_csv(fy_file, index=False)

    # 3. Generate Long Format (CY Series)
    cy_timeseries = preprocess_wpi_timeseries(raw_df, series_type="CY")
    cy_file = target_dir / "agricultural_inputs_wpi_cy_timeseries.csv"
    cy_timeseries.to_csv(cy_file, index=False)

    # 4. Generate Wide Format (FY Series)
    fy_wide = preprocess_wpi_wide(raw_df, series_type="FY")
    wide_file = target_dir / "agricultural_inputs_wpi_fy_wide.csv"
    fy_wide.to_csv(wide_file, index=False)

    return fy_file, cy_file, wide_file, audit_file


def audit_extended_data(df: Optional[pd.DataFrame] = None) -> PreprocessingAudit:
    """Perform schema, nullness, duplicate, and distribution audit of the extended WPI data (FY 2012-13 to 2025-26)."""
    if df is None:
        df = load_staged_wpi_data()

    raw_total = len(df)
    dup_codes = int(df["COMM_CODE"].duplicated().sum())
    dup_names = int(df["COMM_NAME"].duplicated().sum())

    agri_df = get_extended_agricultural_inputs_df(df)
    retained_count = len(agri_df)
    excluded_count = raw_total - retained_count
    categories_breakdown = agri_df["input_category"].value_counts().to_dict()

    fy_obs = int(agri_df[EXTENDED_FY_COLUMNS].notna().sum().sum())

    all_fy_values = agri_df[EXTENDED_FY_COLUMNS].values.flatten()
    clean_fy = all_fy_values[~np.isnan(all_fy_values)]

    min_val = float(np.min(clean_fy)) if len(clean_fy) > 0 else 0.0
    max_val = float(np.max(clean_fy)) if len(clean_fy) > 0 else 0.0
    mean_val = float(np.mean(clean_fy)) if len(clean_fy) > 0 else 0.0

    return PreprocessingAudit(
        raw_total_records=raw_total,
        raw_columns_count=len(df.columns),
        duplicate_commodity_codes=dup_codes,
        duplicate_commodity_names=dup_names,
        unpopulated_columns=[],
        unpopulated_missing_count=0,
        retained_input_commodities=retained_count,
        excluded_commodities_count=excluded_count,
        categories_breakdown=categories_breakdown,
        fy_periods_count=len(EXTENDED_FY_COLUMNS),
        fy_valid_observations=fy_obs,
        cy_periods_count=0,
        cy_valid_observations=0,
        min_wpi_value_fy=round(min_val, 2),
        max_wpi_value_fy=round(max_val, 2),
        mean_wpi_value_fy=round(mean_val, 2),
        target_unit_label=TARGET_LABEL,
        dataset_source="Office of the Economic Adviser, DPIIT, Ministry of Commerce & Industry, Base 2011-12=100 (Extended through FY 2025-26)",
    )


def preprocess_extended_wpi_timeseries(
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract, filter, and reshape extended agricultural input data into a clean, chronologically sorted time-series.

    Spans all 14 Financial Years (FY 2012-13 through FY 2025-26) across 26 agricultural input commodities (364 rows).

    Args:
        df: Staged WPI DataFrame. If None, loaded via load_staged_wpi_data().

    Returns:
        pd.DataFrame: Long-format time series dataset with 364 rows and 16 columns.
    """
    if df is None:
        df = load_staged_wpi_data()

    agri_df = get_extended_agricultural_inputs_df(df)

    rows = []
    for _, item in agri_df.iterrows():
        code = int(item["COMM_CODE"])
        name = str(item["COMM_NAME"])
        weight = float(item["COMM_WT"])
        category = str(item["input_category"])
        suggested_unit = str(item["suggested_unit"])
        description = str(item["description"])

        for order, col in enumerate(EXTENDED_FY_COLUMNS):
            val = item[col]
            if pd.isna(val):
                continue

            float_val = float(val)
            if float_val <= 0:
                raise ValueError(
                    f"Invalid non-positive WPI value {float_val} for {name} ({code}) in {col}."
                )

            lbl = EXTENDED_FY_LABELS[col]
            s_yr = int(lbl.split("-")[0])
            e_yr = int("20" + lbl.split("-")[1])

            rows.append(
                {
                    "commodity_code": code,
                    "commodity_name": name,
                    "commodity_weight": weight,
                    "input_category": category,
                    "suggested_unit": suggested_unit,
                    "description": description,
                    "series_type": "FY",
                    "period_key": col,
                    "period_label": lbl,
                    "period_order": order,
                    "start_year": s_yr,
                    "end_year": e_yr,
                    "wpi_index_points": round(float_val, 2),
                    "target_unit": TARGET_LABEL,
                    "base_year": BASE_YEAR_LABEL,
                    "data_source": "DPIIT, Ministry of Commerce & Industry, Govt of India",
                }
            )

    processed_df = pd.DataFrame(rows)
    processed_df.sort_values(
        by=["commodity_code", "period_order"],
        ascending=[True, True],
        inplace=True,
    )
    processed_df.reset_index(drop=True, inplace=True)
    return processed_df


def preprocess_extended_wpi_wide(
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create a clean wide-format DataFrame for agricultural inputs with all 14 FY periods (26 rows, 20 columns)."""
    if df is None:
        df = load_staged_wpi_data()

    agri_df = get_extended_agricultural_inputs_df(df)

    meta_cols = [
        "COMM_CODE",
        "COMM_NAME",
        "COMM_WT",
        "input_category",
        "suggested_unit",
        "description",
    ]
    wide_df = agri_df[meta_cols + EXTENDED_FY_COLUMNS].copy()

    rename_map = {
        "COMM_CODE": "commodity_code",
        "COMM_NAME": "commodity_name",
        "COMM_WT": "commodity_weight",
    }
    wide_df.rename(columns=rename_map, inplace=True)
    wide_df.sort_values(by="commodity_code", inplace=True)
    wide_df.reset_index(drop=True, inplace=True)
    return wide_df


def save_extended_processed_data(
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """Execute extended preprocessing and save artifacts to data/processed/.

    Saves:
        1. agricultural_inputs_wpi_fy_timeseries_extended.csv (14 FY periods, 364 rows)
        2. agricultural_inputs_wpi_fy_wide_extended.csv (wide format, 26 rows, 20 columns)
        3. extended_preprocessing_audit.json

    Returns:
        Tuple of Paths to the 3 created artifacts.
    """
    target_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    staged_df = load_staged_wpi_data()

    # 1. Audit
    audit = audit_extended_data(staged_df)
    audit_file = target_dir / "extended_preprocessing_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, indent=2)

    # 2. Long Timeseries
    ts_df = preprocess_extended_wpi_timeseries(staged_df)
    ts_file = target_dir / "agricultural_inputs_wpi_fy_timeseries_extended.csv"
    ts_df.to_csv(ts_file, index=False)

    # 3. Wide Format
    wide_df = preprocess_extended_wpi_wide(staged_df)
    wide_file = target_dir / "agricultural_inputs_wpi_fy_wide_extended.csv"
    wide_df.to_csv(wide_file, index=False)

    return ts_file, wide_file, audit_file


if __name__ == "__main__":
    # When executed as script, run preprocessing and display summary
    fy_p, cy_p, wide_p, audit_p = save_processed_data()
    print("Stage 3 Preprocessing Complete.")
    print(f"FY Timeseries: {fy_p}")
    print(f"CY Timeseries: {cy_p}")
    print(f"FY Wide: {wide_p}")
    print(f"Audit: {audit_p}")

    ext_ts, ext_wide, ext_audit = save_extended_processed_data()
    print("Phase 3A Extended Preprocessing Complete.")
    print(f"Extended FY Timeseries: {ext_ts}")
    print(f"Extended FY Wide: {ext_wide}")
    print(f"Extended Audit: {ext_audit}")
