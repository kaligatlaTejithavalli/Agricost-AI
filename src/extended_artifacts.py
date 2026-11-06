"""Extended Production Forecasting & SHAP Artifact Generation for AgriCost-AI.

Phase 3G Artifact Generator:
1. Reuses the trained production candidate pipeline from:
   models/trained/production_candidate_extended.joblib
   WITHOUT retraining or modifying weights.
2. Generates forward forecasts across all 26 commodities for horizons:
   - H1: FY 2026–27
   - H2: FY 2027–28
   - H3: FY 2028–29
   using the extended historical dataset (data/processed/agricultural_inputs_wpi_fy_timeseries_extended.csv).
3. Produces dual trajectories:
   - Ridge ML Trend Forecast (recursive multi-year)
   - Naïve Persistence Static Baseline (last observed FY 2025–26 WPI held constant)
4. Saves forecast table (78 rows) to:
   models/evaluation/forecasts/commodity_forecasts_extended_2026_2029.csv
5. Generates extended SHAP explanations across all 312 observations in:
   data/processed/agricultural_inputs_features_extended.csv
   using authentic shap.LinearExplainer and an Independent background masker.
6. Saves extended SHAP artifacts to:
   models/evaluation/shap/extended/
   - shap_global_importance.csv
   - summary_beeswarm.png
   - summary_bar.png
   - local_explanation_urea_2025_26.json
   - waterfall_urea_2025_26.png
   - shap_assumptions_and_limitations.md
   - shap_metadata_extended.json
7. Numerically verifies exact SHAP additivity (E[f(X)] + sum(phi_i) == y_pred)
   across all 312 samples and specifically for Urea FY 2025–26.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_DIR,
    BASE_YEAR_LABEL,
    EXTENDED_FEATURES_CSV,
    EXTENDED_FORECASTS_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_SHAP_DIR,
    FORECASTS_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.explainability import (
    SHAP_CORRELATED_FEATURES_DISCLAIMER,
    SHAP_UNIT_LABEL,
    SHAP_WPI_VS_RUPEES_DISCLAIMER,
    clean_feature_name_for_display,
    extract_pipeline_components,
    get_transformed_feature_names,
)
from src.feature_engineering import TARGET_COLUMN
from src.forecast import forecast_all_commodities, forecast_commodity

PRODUCTION_CANDIDATE_MODEL_PATH = TRAINED_MODELS_DIR / "production_candidate_extended.joblib"
MODEL_NAME_EXTENDED = "Ridge Regression (alpha=10.0) [Production Candidate Extended]"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hexadecimal digest for a given file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 1. Extended Forward Forecasting (FY 2026-27 through FY 2028-29)
# ---------------------------------------------------------------------------
def generate_extended_forecasts(
    model_path: Optional[Path] = None,
    timeseries_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    forecast_horizon: int = 3,
) -> pd.DataFrame:
    """Generate forward forecasts for all 26 commodities from the FY 2025-26 cutoff.

    Horizons:
    - H1: FY 2026-27
    - H2: FY 2027-28
    - H3: FY 2028-29

    Dual Trajectories:
    - Ridge ML Trend Forecast (forecast_wpi)
    - Naïve Persistence Static Baseline (static_baseline_wpi = FY 2025-26 actual)

    Returns:
        pd.DataFrame: 78 rows (26 commodities * 3 horizons).
    """
    m_path = Path(model_path) if model_path else PRODUCTION_CANDIDATE_MODEL_PATH
    ts_path = Path(timeseries_path) if timeseries_path else EXTENDED_PROCESSED_TIMESERIES_CSV
    out_csv = Path(output_csv_path) if output_csv_path else EXTENDED_FORECASTS_CSV

    if not m_path.exists():
        raise FileNotFoundError(f"Production candidate model not found at: {m_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Extended timeseries not found at: {ts_path}")

    pipeline: Pipeline = joblib.load(m_path)

    # Generate forecasts for all 26 commodities
    forecast_df = forecast_all_commodities(
        forecast_horizon=forecast_horizon,
        pipeline=pipeline,
        timeseries_path=ts_path,
    )

    # Ensure explicit extended model naming
    forecast_df["model_name"] = MODEL_NAME_EXTENDED

    # Validation assertions
    if len(forecast_df) != 78:
        raise ValueError(f"Expected exactly 78 forecast rows, got {len(forecast_df)}.")

    expected_commodities = 26
    if forecast_df["commodity_code"].nunique() != expected_commodities:
        raise ValueError(
            f"Expected {expected_commodities} unique commodities, found {forecast_df['commodity_code'].nunique()}."
        )

    expected_periods = ["2026-27", "2027-28", "2028-29"]
    actual_periods = sorted(forecast_df["forecast_period"].unique().tolist())
    if actual_periods != expected_periods:
        raise ValueError(f"Expected periods {expected_periods}, found {actual_periods}.")

    # Verify no null or non-finite values
    if forecast_df[["forecast_wpi", "static_baseline_wpi", "last_observed_wpi"]].isna().any().any():
        raise ValueError("Missing or NaN values found in forecast dataframe.")

    # Verify static baseline is flat across horizons for each commodity
    for code, group in forecast_df.groupby("commodity_code"):
        unique_baselines = group["static_baseline_wpi"].nunique()
        if unique_baselines != 1:
            raise ValueError(f"Commodity {code} has inconsistent static baseline: {group['static_baseline_wpi'].tolist()}")
        if group["static_baseline_wpi"].iloc[0] != group["last_observed_wpi"].iloc[0]:
            raise ValueError(f"Commodity {code} static baseline does not match last_observed_wpi.")

    # Ensure output directory exists and save artifact
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(out_csv, index=False)

    return forecast_df


# ---------------------------------------------------------------------------
# 2. Extended SHAP Explainability & Additivity Verification
# ---------------------------------------------------------------------------
def generate_extended_shap_artifacts(
    model_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate SHAP explanations and verification artifacts for the extended production candidate.

    Evaluates across all 312 observations in agricultural_inputs_features_extended.csv.
    Produces:
    - Global feature importance table (CSV)
    - Beeswarm & Bar summary plots (PNG)
    - Local explanation JSON & Waterfall plot for Urea FY 2025-26
    - Assumptions & Limitations markdown
    - Metadata JSON with SHA-256 hashes and additivity proof

    Returns:
        Dict[str, Any] with artifacts, summary statistics, and additivity verification.
    """
    m_path = Path(model_path) if model_path else PRODUCTION_CANDIDATE_MODEL_PATH
    feat_path = Path(features_path) if features_path else EXTENDED_FEATURES_CSV
    out_dir = Path(output_dir) if output_dir else EXTENDED_SHAP_DIR

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {m_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Extended features dataset not found at: {feat_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute checksums
    model_sha256 = compute_file_sha256(m_path)
    dataset_sha256 = compute_file_sha256(feat_path)

    # Load pipeline and data
    pipeline: Pipeline = joblib.load(m_path)
    preprocessor, regressor = extract_pipeline_components(pipeline)
    df = pd.read_csv(feat_path)

    if len(df) != 312:
        raise ValueError(f"Expected 312 extended feature rows, found {len(df)}.")

    # Transform feature matrix strictly with fitted preprocessor
    feature_names = get_transformed_feature_names(preprocessor)
    X_trans = preprocessor.transform(df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=np.float64)

    # Create authentic LinearExplainer with Independent background masker
    masker = shap.maskers.Independent(X_trans, max_samples=len(X_trans))
    explainer = shap.LinearExplainer(
        model=regressor,
        masker=masker,
        feature_names=feature_names,
    )

    # Compute SHAP explanation
    shap_explanation = explainer(X_trans)

    # 1. Global Feature Importance
    mean_abs_shap = np.mean(np.abs(shap_explanation.values), axis=0)
    mean_shap = np.mean(shap_explanation.values, axis=0)
    std_shap = np.std(shap_explanation.values, axis=0)

    importance_records = []
    for idx, name in enumerate(feature_names):
        importance_records.append({
            "feature_raw": name,
            "feature_display": clean_feature_name_for_display(name),
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            "mean_shap": round(float(mean_shap[idx]), 4),
            "std_shap": round(float(std_shap[idx]), 4),
            "unit": SHAP_UNIT_LABEL,
        })

    importance_df = pd.DataFrame(importance_records)
    importance_df = importance_df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df = importance_df[["rank", "feature_display", "mean_abs_shap", "mean_shap", "std_shap", "unit", "feature_raw"]]

    csv_path = out_dir / "shap_global_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # 2. Global Beeswarm & Bar Visualizations
    beeswarm_path = out_dir / "summary_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_explanation, max_display=12, show=False)
    plt.title(f"Extended Model SHAP Feature Attribution Distribution ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.xlabel(f"SHAP Value ({SHAP_UNIT_LABEL})", fontsize=10)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    bar_path = out_dir / "summary_bar.png"
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_explanation, max_display=12, show=False)
    plt.title(f"Extended Model Mean Absolute SHAP Feature Importance ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.xlabel(f"Mean |SHAP Value| ({SHAP_UNIT_LABEL})", fontsize=10)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    # 3. Local Explanation for Urea FY 2025-26
    urea_code = 1310020001
    urea_period = "2025-26"
    matches = df[(df["commodity_code"] == urea_code) & (df["period_label"] == urea_period)]
    if matches.empty:
        raise ValueError(f"Urea record for {urea_period} not found in extended features dataset.")

    urea_row_idx = matches.index[0]
    urea_row = df.loc[urea_row_idx]
    urea_actual = float(urea_row[TARGET_COLUMN])
    urea_pred = float(pipeline.predict(df.loc[[urea_row_idx]])[0])
    urea_base = float(shap_explanation.base_values[urea_row_idx])
    urea_contributions = shap_explanation.values[urea_row_idx]

    feature_details = []
    for f_idx, f_name in enumerate(feature_names):
        val = float(urea_contributions[f_idx])
        feature_details.append({
            "feature_raw": f_name,
            "feature_display": clean_feature_name_for_display(f_name),
            "shap_value": round(val, 4),
            "unit": SHAP_UNIT_LABEL,
        })
    feature_details.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    urea_sum_shap = float(np.sum(urea_contributions))
    urea_reconstructed = urea_base + urea_sum_shap
    urea_abs_diff = abs(urea_reconstructed - urea_pred)
    urea_additive = bool(urea_abs_diff <= 1e-4)

    local_urea_json = {
        "commodity_code": int(urea_code),
        "commodity_name": str(urea_row.get("commodity_name", "Urea")),
        "input_category": str(urea_row.get("input_category", "Fertilizers")),
        "period_label": urea_period,
        "sample_index": int(urea_row_idx),
        "actual_wpi": round(urea_actual, 2),
        "predicted_wpi": round(urea_pred, 2),
        "base_value": round(urea_base, 2),
        "sum_of_shap": round(urea_sum_shap, 2),
        "reconstructed_prediction": round(urea_reconstructed, 2),
        "absolute_difference": round(urea_abs_diff, 8),
        "additivity_verified": urea_additive,
        "unit": SHAP_UNIT_LABEL,
        "feature_contributions": feature_details,
        "disclaimer_correlated_features": SHAP_CORRELATED_FEATURES_DISCLAIMER,
        "disclaimer_units": SHAP_WPI_VS_RUPEES_DISCLAIMER,
    }

    urea_json_path = out_dir / "local_explanation_urea_2025_26.json"
    with open(urea_json_path, "w", encoding="utf-8") as f:
        json.dump(local_urea_json, f, indent=2)

    # 4. Local Waterfall Visualization for Urea FY 2025-26
    waterfall_path = out_dir / "waterfall_urea_2025_26.png"
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[urea_row_idx], max_display=12, show=False)
    plt.title(f"Local SHAP Waterfall: Urea FY 2025-26 ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    # 5. Full Additivity Verification across all 312 observations
    all_preds = pipeline.predict(df)
    all_reconstructed = shap_explanation.base_values + np.sum(shap_explanation.values, axis=1)
    all_diffs = np.abs(all_preds - all_reconstructed)
    max_additivity_error = float(np.max(all_diffs))
    all_additive = bool(max_additivity_error < 1e-4)

    # 6. Assumptions and Limitations Documentation
    doc_path = out_dir / "shap_assumptions_and_limitations.md"
    doc_content = (
        "# Extended Explainable AI (SHAP) Assumptions, Mechanics, and Verification\n\n"
        "**Project:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System  \n"
        "**Stage:** Phase 3G — Extended Production Explainability  \n"
        f"**Target Variable:** `wpi_index_points` ({SHAP_UNIT_LABEL})  \n"
        f"**Base Year:** {BASE_YEAR_LABEL}  \n"
        f"**Model Explained:** {MODEL_NAME_EXTENDED}  \n"
        f"**Model Artifact:** `{m_path.as_posix()}` (SHA-256: `{model_sha256}`)  \n"
        f"**Feature Matrix:** `{feat_path.as_posix()}` (SHA-256: `{dataset_sha256}`)  \n"
        "**Observations Explained:** 312 observations across 26 agricultural input commodities  \n"
        "**Temporal Coverage:** Financial Years 2014-15 through 2025-26 (12 consecutive financial years)  \n\n"
        "---\n\n"
        "## 1. Strict Unit Separation & Anti-Rupee Guardrail\n\n"
        f"- **All SHAP feature attributions and base values are strictly in `{SHAP_UNIT_LABEL}`.**\n"
        "- **Never interpret SHAP values as rupee cost variations, retail market price changes, or farmer expenditure adjustments.**\n"
        "- SHAP explains the mathematical mechanics of the WPI index forecasting pipeline. Translation from wholesale index points to physical rupee expenditure is conducted exclusively downstream by the ratio bridge formula in `src/budget.py`:\n\n"
        "  $$\\text{Estimated Future Price (₹)} = \\text{Current User Price (₹)} \\times \\left(\\frac{\\text{Forecast WPI}}{\\text{Current WPI}}\\right)$$\n\n"
        "---\n\n"
        "## 2. Explainer Architecture & Mathematical Grounding\n\n"
        "- **Explainer Implementation:** `shap.LinearExplainer`\n"
        "- **Background Masker:** `shap.maskers.Independent` across all 312 transformed training samples ($N=312$).\n"
        f"- **Expected Value (Base Value):** $E[f(X)] = {np.mean(shap_explanation.base_values):.4f}$ WPI index points.\n"
        "- **Exact Additivity Verification:** Because Ridge Regression is a regularized linear model, Shapley values satisfy exact additivity:\n\n"
        "  $$E[f(X)] + \\sum_{i=1}^{M} \\phi_i(x) = f(x) = \\hat{y}$$\n\n"
        f"- Across all 312 samples, the maximum observed additivity error is **{max_additivity_error:.2e} WPI index points**, verifying exact analytical consistency.\n\n"
        "---\n\n"
        "## 3. Correlated Features & Attribution Dynamics\n\n"
        "The model incorporates 12 transformed predictors (5 numerical time-series indicators + 7 one-hot category indicators):\n"
        "1. `lag_1`: Prior year's WPI index ($y_{t-1}$)\n"
        "2. `lag_2`: Two years prior WPI index ($y_{t-2}$)\n"
        "3. `rolling_mean_2`: Two-year moving average ($(y_{t-1} + y_{t-2}) / 2$)\n"
        "4. `historical_pct_change`: Prior year's percentage index change ($((y_{t-1} - y_{t-2}) / y_{t-2}) \\times 100$)\n"
        "5. `period_order`: Annual chronological trend order index (2 for FY 2014-15 to 13 for FY 2025-26)\n"
        "6. `input_category`: One-hot encoded category indicators with reference category `Farm Machinery`\n\n"
        "### Critical Collinearity Caveat:\n"
        "- Predictors `lag_1`, `lag_2`, `rolling_mean_2`, and `historical_pct_change` exhibit strong collinearity.\n"
        "- Under `shap.LinearExplainer`, attributions reflect model coefficient weighting and marginal distributions rather than independent causal levers.\n"
        "- A feature contribution indicates how the model adjusted its projection given that feature's value relative to the training mean, **NOT a real-world causal guarantee**.\n"
    )
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)

    # 7. Extended Metadata JSON
    top_5_features = importance_df.head(5).to_dict(orient="records")
    metadata = {
        "model_name": MODEL_NAME_EXTENDED,
        "model_artifact": str(m_path.as_posix()),
        "model_sha256": model_sha256,
        "explainer_type": "shap.LinearExplainer",
        "masker_type": "shap.maskers.Independent",
        "feature_dataset": str(feat_path.as_posix()),
        "feature_dataset_sha256": dataset_sha256,
        "sample_count": len(df),
        "commodity_count": int(df["commodity_code"].nunique()),
        "first_period": "2014-15",
        "last_period": "2025-26",
        "explanation_periods": sorted(df["period_label"].unique().tolist()),
        "num_features_explained": len(feature_names),
        "target_variable": TARGET_COLUMN,
        "target_unit": SHAP_UNIT_LABEL,
        "base_value_expected": round(float(np.mean(shap_explanation.base_values)), 4),
        "additivity_verified_all_samples": all_additive,
        "max_additivity_error": max_additivity_error,
        "top_5_features_by_mean_abs_shap": top_5_features,
        "local_example_urea_2025_26": {
            "commodity_code": urea_code,
            "commodity_name": local_urea_json["commodity_name"],
            "period": urea_period,
            "actual_wpi": local_urea_json["actual_wpi"],
            "predicted_wpi": local_urea_json["predicted_wpi"],
            "base_value": local_urea_json["base_value"],
            "sum_of_shap": local_urea_json["sum_of_shap"],
            "reconstructed": local_urea_json["reconstructed_prediction"],
            "absolute_difference": local_urea_json["absolute_difference"],
            "additivity_verified": local_urea_json["additivity_verified"],
            "unit": SHAP_UNIT_LABEL,
        },
        "artifacts_saved": {
            "global_importance_csv": str(csv_path.as_posix()),
            "summary_beeswarm_plot": str(beeswarm_path.as_posix()),
            "summary_bar_plot": str(bar_path.as_posix()),
            "local_explanation_json": str(urea_json_path.as_posix()),
            "waterfall_plot": str(waterfall_path.as_posix()),
            "assumptions_markdown": str(doc_path.as_posix()),
        },
    }

    meta_path = out_dir / "shap_metadata_extended.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "metadata": metadata,
        "importance_df": importance_df,
        "local_urea": local_urea_json,
        "artifacts": metadata["artifacts_saved"],
        "max_additivity_error": max_additivity_error,
        "additivity_verified": all_additive,
    }


def run_all_extended_artifacts() -> Dict[str, Any]:
    """Execute both extended forward forecasting and SHAP artifact generation."""
    print("=" * 80)
    print("PHASE 3G: GENERATING EXTENDED PRODUCTION FORECASTS & SHAP ARTIFACTS")
    print("=" * 80)

    # 1. Forecasts
    print("\n1. Generating Extended Forward Forecasts (FY 2026-27 to FY 2028-29)...")
    forecast_df = generate_extended_forecasts()
    print(f"   Successfully generated {len(forecast_df)} forecast records across {forecast_df['commodity_code'].nunique()} commodities.")
    print(f"   Saved artifact: {EXTENDED_FORECASTS_CSV}")

    # 2. SHAP
    print("\n2. Generating Extended SHAP Explanations & Visualizations...")
    shap_results = generate_extended_shap_artifacts()
    print(f"   Max Additivity Error across {shap_results['metadata']['sample_count']} samples: {shap_results['max_additivity_error']:.2e} WPI index points.")
    print(f"   All samples additive: {shap_results['additivity_verified']}")
    print(f"   Saved SHAP artifacts under: {EXTENDED_SHAP_DIR}")

    return {
        "forecast_df": forecast_df,
        "shap_results": shap_results,
    }


if __name__ == "__main__":
    run_all_extended_artifacts()
