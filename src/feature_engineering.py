"""Feature engineering and chronological train/test splitting module for AgriCost-AI.

Stage 4 Feature Engineering Module:
1. Ingests clean longitudinal time-series data from data/processed/.
2. Constructs leak-free historical lag and momentum features strictly using past observations:
   - lag_1: Prior year's WPI index value (y_{t-1})
   - lag_2: Two years prior WPI index value (y_{t-2})
   - historical_pct_change: Prior year's percentage index movement ((lag_1 - lag_2) / lag_2) * 100
   - rolling_mean_2: Two-year past rolling average ((lag_1 + lag_2) / 2)
   - period_order: Integer time trend index
3. Prevents cross-commodity contamination by strictly computing shifts and rolling metrics
   within individual commodity groups.
4. Enforces strict chronological train/test splitting (Train: 2014-15 to 2019-20; Test: 2020-21 to 2022-23)
   with zero data leakage and zero random shuffling.
5. Preserves all WPI index values strictly as 'WPI index points' (never rupees).
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.config import (
    EXTENDED_FEATURES_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    PROCESSED_DATA_DIR,
    TARGET_LABEL,
)

# Standard feature set justified by the 11-period annual series
CORE_FEATURE_COLUMNS: List[str] = [
    "lag_1",
    "lag_2",
    "historical_pct_change",
    "rolling_mean_2",
    "period_order",
]

TARGET_COLUMN: str = "wpi_index_points"

# Chronological split definitions
DEFAULT_TRAIN_PERIODS: List[str] = [
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
]

DEFAULT_TEST_PERIODS: List[str] = [
    "2020-21",
    "2021-22",
    "2022-23",
]


@dataclass(frozen=True)
class FeatureEngineeringSummary:
    """Summary metrics of the feature engineering and chronological split pipeline."""

    total_commodities: int
    raw_timeseries_rows: int
    features_created: List[str]
    dropped_initial_lag_rows: int
    usable_rows: int
    usable_observations_per_commodity: int
    train_periods: List[str]
    train_samples_count: int
    test_periods: List[str]
    test_samples_count: int
    target_variable: str
    target_unit_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_processed_timeseries(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the preprocessed Financial Year timeseries from data/processed/.

    Args:
        filepath: Optional path to the CSV file. If None, uses default FY timeseries path.

    Returns:
        pd.DataFrame: Clean timeseries DataFrame.

    Raises:
        FileNotFoundError: If the processed timeseries file does not exist.
    """
    path = filepath if filepath else PROCESSED_DATA_DIR / "agricultural_inputs_wpi_fy_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed timeseries not found at {path}. Please run Stage 3 preprocessing first."
        )

    df = pd.read_csv(path)
    return df


def create_time_series_features(
    df: Optional[pd.DataFrame] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Construct leak-free chronological features for each agricultural input commodity.

    CRITICAL LEAKAGE PREVENTION:
    1. Grouping: Calculations are grouped strictly by 'commodity_code'. An observation for
       one commodity will NEVER use or shift values from an adjacent commodity.
    2. Zero Future Information: All features for year T use strictly past observations (t-1, t-2).
       Target y_T is never included in the historical percentage change or rolling statistics.
    3. Monotonic Ordering: The data is sorted by commodity_code and period_order prior to calculation.

    Features Created:
    - lag_1: y_{t-1} (Previous year's observed WPI index points)
    - lag_2: y_{t-2} (WPI index points observed 2 years prior)
    - historical_pct_change: ((lag_1 - lag_2) / lag_2) * 100 (Prior year's momentum/inflation rate)
    - rolling_mean_2: (lag_1 + lag_2) / 2.0 (Two-year past rolling average)
    - period_order: Integer sequence index (0 to 10) representing linear macro time trend.

    Args:
        df: Input timeseries DataFrame. If None, loads from data/processed/.
        drop_na: If True, drops initial periods (t=0 and t=1) where lag_1 or lag_2 are unavailable.

    Returns:
        pd.DataFrame: DataFrame containing original metadata, features, and target variable.
    """
    if df is None:
        df = load_processed_timeseries()

    # Enforce strict sort order: commodity first, then chronological period_order
    feature_df = df.sort_values(
        by=["commodity_code", "period_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # 1. Grouped lag features (strictly within each commodity)
    feature_df["lag_1"] = feature_df.groupby("commodity_code")[TARGET_COLUMN].shift(1)
    feature_df["lag_2"] = feature_df.groupby("commodity_code")[TARGET_COLUMN].shift(2)

    # 2. Historical momentum / percentage change from t-2 to t-1
    # Uses ONLY past observations (lag_1 and lag_2); NEVER uses current year t's target
    feature_df["historical_pct_change"] = (
        (feature_df["lag_1"] - feature_df["lag_2"]) / feature_df["lag_2"]
    ) * 100.0

    # 3. Two-year past rolling average
    feature_df["rolling_mean_2"] = (feature_df["lag_1"] + feature_df["lag_2"]) / 2.0

    # 4. Round feature columns for numerical consistency
    feature_df["lag_1"] = feature_df["lag_1"].round(2)
    feature_df["lag_2"] = feature_df["lag_2"].round(2)
    feature_df["historical_pct_change"] = feature_df["historical_pct_change"].round(4)
    feature_df["rolling_mean_2"] = feature_df["rolling_mean_2"].round(2)

    # 5. Handle missing lag values transparently
    if drop_na:
        feature_df = feature_df.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)

    return feature_df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_periods: Optional[List[str]] = None,
    test_periods: Optional[List[str]] = None,
    commodity_code: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train/test split on feature-engineered data.

    CRITICAL RULES:
    1. Zero Random Shuffling: Chronological integrity is strictly preserved.
    2. Temporal Separation: Training periods must strictly precede testing periods.
       Default split:
         - Training: 2014-15 through 2019-20 (6 annual periods, pre-COVID baseline)
         - Testing: 2020-21 through 2022-23 (3 annual periods, held-out validation)
    3. Commodity Filtering: Optionally isolates a single commodity while maintaining
       identical temporal boundaries.

    Args:
        df: Feature-engineered DataFrame from create_time_series_features(drop_na=True).
        train_periods: List of period labels for training. Defaults to DEFAULT_TRAIN_PERIODS.
        test_periods: List of period labels for testing. Defaults to DEFAULT_TEST_PERIODS.
        commodity_code: Optional commodity code to extract train/test sets for a single commodity.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)

    Raises:
        ValueError: If periods overlap, are missing from the data, or result in empty splits.
    """
    tr_periods = list(train_periods) if train_periods is not None else DEFAULT_TRAIN_PERIODS
    te_periods = list(test_periods) if test_periods is not None else DEFAULT_TEST_PERIODS

    # Validate disjoint periods (no overlap between train and test)
    overlap = set(tr_periods).intersection(set(te_periods))
    if overlap:
        raise ValueError(
            f"Chronological split error: Overlapping periods found in train and test sets: {overlap}."
        )

    working_df = df
    if commodity_code is not None:
        working_df = df[df["commodity_code"] == int(commodity_code)].copy()
        if working_df.empty:
            raise ValueError(f"Commodity code {commodity_code} not found in feature dataset.")

    available_periods = set(working_df["period_label"].unique())
    missing_train = set(tr_periods) - available_periods
    if missing_train:
        raise ValueError(f"Requested train periods not found in dataset: {missing_train}")
    missing_test = set(te_periods) - available_periods
    if missing_test:
        raise ValueError(f"Requested test periods not found in dataset: {missing_test}")

    train_df = working_df[working_df["period_label"].isin(tr_periods)].copy()
    test_df = working_df[working_df["period_label"].isin(te_periods)].copy()

    # Enforce chronological ordering
    train_df.sort_values(by=["commodity_code", "period_order"], inplace=True)
    test_df.sort_values(by=["commodity_code", "period_order"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Validate that maximum training period order is strictly less than minimum test period order
    if not train_df.empty and not test_df.empty:
        max_tr_order = train_df["period_order"].max()
        min_te_order = test_df["period_order"].min()
        if max_tr_order >= min_te_order:
            raise ValueError(
                f"Chronological boundary violation: Max train period_order ({max_tr_order}) "
                f"is >= min test period_order ({min_te_order}). Train must strictly precede test."
            )

    return train_df, test_df


def get_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extract standard scikit-learn compatible feature matrices (X) and target vectors (y).

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.
        feature_cols: List of feature column names. Defaults to CORE_FEATURE_COLUMNS.
        target_col: Target column name. Defaults to TARGET_COLUMN ('wpi_index_points').

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            (X_train, y_train, X_test, y_test)
    """
    cols = list(feature_cols) if feature_cols is not None else CORE_FEATURE_COLUMNS

    missing_train = [c for c in cols if c not in train_df.columns]
    if missing_train:
        raise ValueError(f"Feature columns missing from train_df: {missing_train}")

    missing_test = [c for c in cols if c not in test_df.columns]
    if missing_test:
        raise ValueError(f"Feature columns missing from test_df: {missing_test}")

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from datasets.")

    X_train = train_df[cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[cols].copy()
    y_test = test_df[target_col].copy()

    return X_train, y_train, X_test, y_test


def save_feature_engineered_datasets(
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Path]:
    """Execute complete feature engineering pipeline and persist datasets to data/processed/.

    Saves:
        1. agricultural_inputs_features.csv (Complete usable feature dataset, 234 rows)
        2. train_data.csv (Training partition, 156 rows)
        3. test_data.csv (Testing partition, 78 rows)
        4. feature_engineering_summary.json (Metadata, periods, sample counts)

    Args:
        output_dir: Directory where processed files will be saved. Defaults to PROCESSED_DATA_DIR.

    Returns:
        Tuple of Paths to the 4 generated files.
    """
    target_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_ts = load_processed_timeseries()
    feature_df = create_time_series_features(raw_ts, drop_na=True)
    train_df, test_df = chronological_train_test_split(feature_df)

    # 1. Save complete feature dataset
    features_path = target_dir / "agricultural_inputs_features.csv"
    feature_df.to_csv(features_path, index=False)

    # 2. Save train partition
    train_path = target_dir / "train_data.csv"
    train_df.to_csv(train_path, index=False)

    # 3. Save test partition
    test_path = target_dir / "test_data.csv"
    test_df.to_csv(test_path, index=False)

    # 4. Generate and save summary JSON
    summary = FeatureEngineeringSummary(
        total_commodities=int(feature_df["commodity_code"].nunique()),
        raw_timeseries_rows=len(raw_ts),
        features_created=list(CORE_FEATURE_COLUMNS),
        dropped_initial_lag_rows=len(raw_ts) - len(feature_df),
        usable_rows=len(feature_df),
        usable_observations_per_commodity=int(len(feature_df) / feature_df["commodity_code"].nunique()),
        train_periods=list(train_df["period_label"].unique()),
        train_samples_count=len(train_df),
        test_periods=list(test_df["period_label"].unique()),
        test_samples_count=len(test_df),
        target_variable=TARGET_COLUMN,
        target_unit_label=TARGET_LABEL,
    )

    summary_path = target_dir / "feature_engineering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    return features_path, train_path, test_path, summary_path


def load_extended_processed_timeseries(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the extended Financial Year timeseries from data/processed/.

    Args:
        filepath: Optional path to the CSV file. If None, uses EXTENDED_PROCESSED_TIMESERIES_CSV.

    Returns:
        pd.DataFrame: Extended timeseries DataFrame (364 rows across 14 FY periods).

    Raises:
        FileNotFoundError: If the extended timeseries file does not exist.
    """
    path = filepath if filepath else EXTENDED_PROCESSED_TIMESERIES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Extended timeseries not found at {path}. Please run Phase 3A extended preprocessing first."
        )

    return pd.read_csv(path)


def create_extended_time_series_features(
    df: Optional[pd.DataFrame] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Construct leak-free chronological features for the extended timeseries (FY 2012-13 to FY 2025-26).

    CRITICAL LEAKAGE PREVENTION:
    1. Grouping: Calculations are grouped strictly by 'commodity_code'.
    2. Zero Future Information: All features for year T use strictly past observations (t-1, t-2).
    3. Monotonic Ordering: The data is sorted by commodity_code and period_order (0 to 13) prior to calculation.

    Features Created:
    - lag_1: y_{t-1}
    - lag_2: y_{t-2}
    - historical_pct_change: ((lag_1 - lag_2) / lag_2) * 100
    - rolling_mean_2: (lag_1 + lag_2) / 2.0
    - period_order: Integer sequence index (0 to 13).

    Args:
        df: Extended timeseries DataFrame. If None, loads from EXTENDED_PROCESSED_TIMESERIES_CSV.
        drop_na: If True, drops initial periods (t=0 and t=1) where lag_1 or lag_2 are unavailable.

    Returns:
        pd.DataFrame: Extended feature matrix (312 rows when drop_na=True: 26 commodities * 12 usable periods).
    """
    if df is None:
        df = load_extended_processed_timeseries()

    feature_df = df.sort_values(
        by=["commodity_code", "period_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    feature_df["lag_1"] = feature_df.groupby("commodity_code")[TARGET_COLUMN].shift(1)
    feature_df["lag_2"] = feature_df.groupby("commodity_code")[TARGET_COLUMN].shift(2)

    feature_df["historical_pct_change"] = (
        (feature_df["lag_1"] - feature_df["lag_2"]) / feature_df["lag_2"]
    ) * 100.0

    feature_df["rolling_mean_2"] = (feature_df["lag_1"] + feature_df["lag_2"]) / 2.0

    feature_df["lag_1"] = feature_df["lag_1"].round(2)
    feature_df["lag_2"] = feature_df["lag_2"].round(2)
    feature_df["historical_pct_change"] = feature_df["historical_pct_change"].round(4)
    feature_df["rolling_mean_2"] = feature_df["rolling_mean_2"].round(2)

    if drop_na:
        feature_df = feature_df.dropna(subset=["lag_1", "lag_2"]).reset_index(drop=True)

    return feature_df


def save_extended_feature_engineered_datasets(
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Execute extended feature engineering and persist artifacts to data/processed/.

    Saves:
        1. agricultural_inputs_features_extended.csv (312 usable rows, FY 2014-15 to FY 2025-26)
        2. extended_feature_engineering_summary.json

    Returns:
        Tuple of Paths to the 2 generated files.
    """
    target_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_ts = load_extended_processed_timeseries()
    feature_df = create_extended_time_series_features(raw_ts, drop_na=True)

    features_path = target_dir / "agricultural_inputs_features_extended.csv"
    feature_df.to_csv(features_path, index=False)

    summary = {
        "phase": "Phase 3A Extended Historical Data Integration",
        "total_commodities": int(feature_df["commodity_code"].nunique()),
        "raw_timeseries_rows": len(raw_ts),
        "features_created": list(CORE_FEATURE_COLUMNS),
        "dropped_initial_lag_rows": len(raw_ts) - len(feature_df),
        "usable_rows": len(feature_df),
        "usable_observations_per_commodity": int(len(feature_df) / feature_df["commodity_code"].nunique()),
        "first_financial_year": "2012-13",
        "last_financial_year": "2025-26",
        "usable_periods": sorted(list(feature_df["period_label"].unique())),
        "usable_periods_count": int(feature_df["period_label"].nunique()),
        "target_variable": TARGET_COLUMN,
        "target_unit_label": TARGET_LABEL,
        "temporal_leakage_prevented": True,
        "zero_cross_commodity_contamination": True,
    }

    summary_path = target_dir / "extended_feature_engineering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return features_path, summary_path


if __name__ == "__main__":
    feat_p, tr_p, te_p, sum_p = save_feature_engineered_datasets()
    print("Stage 4 Feature Engineering Complete.")
    print(f"Features CSV: {feat_p}")
    print(f"Train CSV:    {tr_p}")
    print(f"Test CSV:     {te_p}")
    print(f"Summary JSON: {sum_p}")

    ext_feat_p, ext_sum_p = save_extended_feature_engineered_datasets()
    print("Phase 3A Extended Feature Engineering Complete.")
    print(f"Extended Features CSV: {ext_feat_p}")
    print(f"Extended Summary JSON: {ext_sum_p}")
