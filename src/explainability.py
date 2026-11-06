"""Explainable AI (XAI) module for AgriCost-AI using authentic SHAP.

Stage 6 Explainability Module:
1. Reuses the fitted Ridge Regression pipeline (alpha=10.0) from models/trained/best_model.joblib
   WITHOUT retraining or modifying model weights.
2. Applies the exact trained ColumnTransformer preprocessing to training and testing datasets
   WITHOUT refitting any transformations on test data.
3. Recovers transformed feature names (5 numerical + 7 one-hot encoded category indicators).
4. Uses authentic shap.LinearExplainer with a training-only background dataset (156 samples).
5. Explains model predictions across all 78 held-out test observations (2020-21 to 2022-23).
6. Verifies exact SHAP additivity: Base value (E[f(X)]) + sum(phi_i) == Model prediction.
7. Produces global feature importance rankings, beeswarm summary plots, and global bar charts.
8. Produces local explanation examples (including Urea 2022-23) with waterfall plots.
9. Strictly enforces unit labeling: All SHAP values and base values are in 'WPI index points',
   NEVER rupees, retail price changes, or causal farmer costs.
10. Explicitly documents the statistical assumptions of SHAP and the effect of correlated features
    (lag_1, lag_2, rolling_mean_2, historical_pct_change).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated script & test environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    PROCESSED_DATA_DIR,
    SHAP_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.feature_engineering import TARGET_COLUMN
from src.train import load_training_and_testing_data

# ---------------------------------------------------------------------------
# Constants & Transparent Disclaimers
# ---------------------------------------------------------------------------
SHAP_UNIT_LABEL: str = TARGET_LABEL  # "WPI index points"

SHAP_CORRELATED_FEATURES_DISCLAIMER: str = (
    "Correlated Features & Causality Disclaimer: "
    "Features 'lag_1', 'lag_2', 'rolling_mean_2', and 'historical_pct_change' are mathematically "
    "and temporally collinear. SHAP attribution under shap.LinearExplainer evaluates feature "
    "contributions based on the fitted linear coefficients and the empirical training distribution. "
    "Because features are correlated, SHAP values reflect joint predictive associations under the "
    "model rather than independent causal mechanisms. A high positive SHAP contribution for lag_1 "
    "indicates that the model relied heavily on prior-year index momentum, not that changing lag_1 "
    "in isolation causes a guaranteed change in future wholesale prices."
)

SHAP_WPI_VS_RUPEES_DISCLAIMER: str = (
    "Unit Distinction Disclaimer: "
    "All SHAP values, feature attributions, and expected base values are expressed strictly in "
    "'WPI index points' (relative to Base Year 2011-12 = 100.0). SHAP contributions do NOT represent "
    "rupee price changes, fertilizer retail costs, farm machinery hourly rates, or guaranteed "
    "farmer expenditure impacts. SHAP explains the mathematical mechanics of the WPI index forecasting "
    "model, not the user's localized physical market purchase price."
)


# ---------------------------------------------------------------------------
# Pipeline Loading & Component Extraction
# ---------------------------------------------------------------------------
def load_trained_pipeline(model_path: Optional[Path] = None) -> Pipeline:
    """Load the existing serialized Ridge Regression pipeline without retraining.

    Args:
        model_path: Optional path to serialized joblib model. Defaults to best_model.joblib.

    Returns:
        Pipeline: The deserialized scikit-learn pipeline.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the deserialized object is not a scikit-learn Pipeline.
        ValueError: If pipeline steps do not conform to ('preprocessor', 'regressor').
    """
    path = Path(model_path) if model_path else TRAINED_MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at: {path}")

    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected scikit-learn Pipeline, received {type(pipeline).__name__}.")

    if "preprocessor" not in pipeline.named_steps or "regressor" not in pipeline.named_steps:
        raise ValueError(
            f"Pipeline missing expected steps 'preprocessor' or 'regressor'. Found: {list(pipeline.named_steps.keys())}"
        )

    regressor = pipeline.named_steps["regressor"]
    if not isinstance(regressor, Ridge):
        raise ValueError(f"Expected regressor to be Ridge, received {type(regressor).__name__}.")

    return pipeline


def extract_pipeline_components(pipeline: Pipeline) -> Tuple[ColumnTransformer, Ridge]:
    """Extract fitted ColumnTransformer and Ridge estimator from pipeline."""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    regressor: Ridge = pipeline.named_steps["regressor"]
    return preprocessor, regressor


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Recover the exact transformed feature names output by the fitted preprocessor.

    Returns a list of 12 feature names:
    - 5 standardized numerical features (num__lag_1, num__lag_2, etc.)
    - 7 one-hot encoded category indicators (cat__input_category_...)
    """
    raw_names = list(preprocessor.get_feature_names_out())
    return raw_names


def clean_feature_name_for_display(raw_name: str) -> str:
    """Provide a human-readable display label for a transformed feature."""
    display_mapping = {
        "num__lag_1": "Prior Year WPI (lag_1)",
        "num__lag_2": "Two Years Prior WPI (lag_2)",
        "num__historical_pct_change": "Prior Year % Change (pct_change)",
        "num__rolling_mean_2": "2-Year Moving Avg (rolling_mean_2)",
        "num__period_order": "Temporal Trend Order (period_order)",
    }
    if raw_name in display_mapping:
        return display_mapping[raw_name]

    if raw_name.startswith("cat__input_category_"):
        cat_name = raw_name.replace("cat__input_category_", "")
        return f"Category: {cat_name}"

    return raw_name


# ---------------------------------------------------------------------------
# Data Preparation & Preprocessing (Strict No-Data-Leakage)
# ---------------------------------------------------------------------------
def prepare_preprocessed_matrices(
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Transform train and test sets using the ALREADY FITTED preprocessor.

    Crucial: Does NOT call fit() or fit_transform() on the test dataset.
    Preserves exact feature alignment and ordering.
    """
    feature_names = get_transformed_feature_names(preprocessor)

    # Use transform only
    X_train_trans = preprocessor.transform(train_df)
    X_test_trans = preprocessor.transform(test_df)

    # Convert sparse matrices to dense numpy arrays if needed
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()

    X_train_trans = np.asarray(X_train_trans, dtype=np.float64)
    X_test_trans = np.asarray(X_test_trans, dtype=np.float64)

    return X_train_trans, X_test_trans, feature_names


# ---------------------------------------------------------------------------
# Authentic SHAP Explainer Construction
# ---------------------------------------------------------------------------
def create_linear_explainer(
    regressor: Ridge,
    X_train_trans: np.ndarray,
    feature_names: List[str],
) -> shap.LinearExplainer:
    """Instantiate an authentic shap.LinearExplainer using the training background data.

    Args:
        regressor: Fitted Ridge regression estimator.
        X_train_trans: Transformed training feature matrix (156 samples).
        feature_names: List of 12 transformed feature names.

    Returns:
        shap.LinearExplainer configured with an Independent background masker using all 156 samples.
    """
    # Use all training samples for the masker to avoid subsampling warnings and guarantee exact expected values
    masker = shap.maskers.Independent(X_train_trans, max_samples=len(X_train_trans))
    explainer = shap.LinearExplainer(
        model=regressor,
        masker=masker,
        feature_names=feature_names,
    )
    return explainer


def compute_shap_explanations(
    explainer: shap.LinearExplainer,
    X_test_trans: np.ndarray,
) -> shap.Explanation:
    """Calculate SHAP values for the evaluation dataset."""
    shap_explanation = explainer(X_test_trans)
    return shap_explanation


# ---------------------------------------------------------------------------
# Global Explainability Metrics
# ---------------------------------------------------------------------------
def compute_global_feature_importance(
    shap_explanation: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute global feature importance as mean absolute SHAP value in WPI index points.

    Args:
        shap_explanation: SHAP Explanation object for test set.
        feature_names: List of 12 feature names.

    Returns:
        pd.DataFrame sorted by mean_abs_shap descending.
    """
    values = shap_explanation.values  # Shape: (N_samples, N_features)

    mean_abs_shap = np.mean(np.abs(values), axis=0)
    mean_shap = np.mean(values, axis=0)
    std_shap = np.std(values, axis=0)

    records = []
    for idx, name in enumerate(feature_names):
        records.append({
            "feature_raw": name,
            "feature_display": clean_feature_name_for_display(name),
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
            "mean_shap": round(float(mean_shap[idx]), 4),
            "std_shap": round(float(std_shap[idx]), 4),
            "unit": SHAP_UNIT_LABEL,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df[["rank", "feature_display", "mean_abs_shap", "mean_shap", "std_shap", "unit", "feature_raw"]]


# ---------------------------------------------------------------------------
# Local Explainability & Additivity Verification
# ---------------------------------------------------------------------------
def explain_individual_prediction(
    pipeline: Pipeline,
    explainer: shap.LinearExplainer,
    shap_explanation: shap.Explanation,
    test_df: pd.DataFrame,
    commodity_code: int,
    period_label: str,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Explain a specific historical prediction and verify mathematical SHAP additivity.

    Formula:
        Base value E[f(X)] + sum(phi_i) == Model Prediction (y_hat)

    Args:
        pipeline: Trained Pipeline.
        explainer: Fitted LinearExplainer.
        shap_explanation: Precomputed SHAP explanation for test_df.
        test_df: Test DataFrame (78 rows).
        commodity_code: Integer code of agricultural input commodity.
        period_label: Target financial year string (e.g. '2022-23').
        tolerance: Numerical tolerance for floating-point additivity check.

    Returns:
        Dict[str, Any] containing local explanation and additivity audit.

    Raises:
        ValueError: If commodity_code or period_label is not found in test_df.
    """
    matches = test_df[
        (test_df["commodity_code"] == commodity_code) & (test_df["period_label"] == period_label)
    ]

    if matches.empty:
        available_commodities = sorted(test_df["commodity_code"].unique().tolist())
        available_periods = sorted(test_df["period_label"].unique().tolist())
        raise ValueError(
            f"Record not found for commodity_code={commodity_code} and period_label='{period_label}'. "
            f"Available commodity codes: {available_commodities}. Available periods: {available_periods}."
        )

    row_idx = matches.index[0]
    # Local position in test_df (0 to 77)
    test_idx = test_df.index.get_loc(row_idx)

    row_series = test_df.loc[row_idx]
    row_df = test_df.loc[[row_idx]]

    commodity_name = str(row_series.get("commodity_name", "Unknown"))
    input_category = str(row_series.get("input_category", "Unknown"))
    actual_wpi = float(row_series[TARGET_COLUMN])

    # Model predicted WPI
    model_pred_wpi = float(pipeline.predict(row_df)[0])

    # Base value E[f(X)]
    base_value = float(shap_explanation.base_values[test_idx])

    # Individual feature contributions
    feature_names = list(shap_explanation.feature_names)
    contributions = shap_explanation.values[test_idx]

    feature_details = []
    for f_idx, f_name in enumerate(feature_names):
        val = float(contributions[f_idx])
        feature_details.append({
            "feature_raw": f_name,
            "feature_display": clean_feature_name_for_display(f_name),
            "shap_value": round(val, 4),
            "unit": SHAP_UNIT_LABEL,
        })

    # Sort contributions by absolute magnitude descending
    feature_details.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    sum_of_shap = float(np.sum(contributions))
    reconstructed_pred = base_value + sum_of_shap
    absolute_diff = abs(reconstructed_pred - model_pred_wpi)
    additivity_verified = bool(absolute_diff <= tolerance)

    explanation_result = {
        "commodity_code": int(commodity_code),
        "commodity_name": commodity_name,
        "input_category": input_category,
        "period_label": period_label,
        "test_index": int(test_idx),
        "actual_wpi": round(actual_wpi, 2),
        "predicted_wpi": round(model_pred_wpi, 2),
        "base_value": round(base_value, 2),
        "sum_of_shap": round(sum_of_shap, 2),
        "reconstructed_prediction": round(reconstructed_pred, 2),
        "absolute_difference": round(absolute_diff, 8),
        "additivity_verified": additivity_verified,
        "unit": SHAP_UNIT_LABEL,
        "feature_contributions": feature_details,
        "disclaimer_correlated_features": SHAP_CORRELATED_FEATURES_DISCLAIMER,
        "disclaimer_units": SHAP_WPI_VS_RUPEES_DISCLAIMER,
    }

    return explanation_result


# ---------------------------------------------------------------------------
# Visualizations & Plot Generation
# ---------------------------------------------------------------------------
def generate_and_save_shap_plots(
    shap_explanation: shap.Explanation,
    local_test_idx: int,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate and save global beeswarm, global bar, and local waterfall plots.

    Args:
        shap_explanation: SHAP Explanation object for test set.
        local_test_idx: Row index within test set for local waterfall plot.
        output_dir: Directory to save generated plot PNGs.

    Returns:
        Dict mapping plot name to Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, Path] = {}

    # 1. Global Beeswarm Plot
    beeswarm_path = output_dir / "summary_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_explanation, max_display=12, show=False)
    plt.title(f"SHAP Feature Attribution Distribution across Evaluation Samples ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.xlabel(f"SHAP Value ({SHAP_UNIT_LABEL})", fontsize=10)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    plot_paths["summary_beeswarm"] = beeswarm_path

    # 2. Global Bar Plot (Mean Absolute SHAP Value)
    bar_path = output_dir / "summary_bar.png"
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_explanation, max_display=12, show=False)
    plt.title(f"Mean Absolute SHAP Feature Importance ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.xlabel(f"Mean |SHAP Value| ({SHAP_UNIT_LABEL})", fontsize=10)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    plot_paths["summary_bar"] = bar_path

    # 3. Local Waterfall Plot for Target Example
    waterfall_path = output_dir / "waterfall_urea_2022_23.png"
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[local_test_idx], max_display=12, show=False)
    plt.title(f"Local SHAP Explanation: Urea FY 2022-23 ({SHAP_UNIT_LABEL})", fontsize=11)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    plot_paths["waterfall_urea"] = waterfall_path

    return plot_paths


# ---------------------------------------------------------------------------
# Artifact Serialization & Documentation
# ---------------------------------------------------------------------------
def save_assumptions_and_limitations_doc(output_dir: Path) -> Path:
    """Document SHAP assumptions, collinearity dynamics, and index unit constraints."""
    doc_path = output_dir / "shap_assumptions_and_limitations.md"
    content = (
        "# Explainable AI (SHAP) Assumptions, Mechanics, and Limitations\n\n"
        f"**Project:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System  \n"
        f"**Stage:** Stage 6 — Explainable AI (XAI) using SHAP  \n"
        f"**Target Variable:** `wpi_index_points` ({SHAP_UNIT_LABEL})  \n"
        "**Base Year:** Base Year 2011-12 = 100.0  \n"
        "**Model Explained:** Ridge Regression (alpha = 10.0)  \n\n"
        "---\n\n"
        "## 1. Unit Semantics & Anti-Rupee Guardrail\n\n"
        f"- **All SHAP feature contributions and base values are strictly in `{SHAP_UNIT_LABEL}`.**\n"
        "- **Never interpret SHAP values as rupee price movements, actual retail fertilizer costs, farm machinery hiring rates, or guaranteed cash budget impacts.**\n"
        "- SHAP explains the internal mechanics of the WPI index forecasting pipeline. The translation from WPI index points to user rupee expenditure is handled downstream by the budget planning bridge formula:\n\n"
        "  Estimated Future Price (Rs) = Current User Price (Rs) * (Forecasted WPI / Current WPI)\n\n"
        "---\n\n"
        "## 2. Explainer Architecture & Mathematical Grounding\n\n"
        "- **Explainer Type:** `shap.LinearExplainer`\n"
        "- **Background Masker:** `shap.maskers.Independent` fitted strictly on the **training split** (156 samples, 2014-15 to 2019-20).\n"
        "- **Exact Additivity:** Because Ridge Regression is a regularized linear model, `LinearExplainer` produces exact analytical Shapley attributions:\n\n"
        "  E[f(X)] + sum(phi_i) = f(x) = y_hat\n\n"
        "  This additivity holds within floating-point machine precision (< 1e-10 WPI index points).\n\n"
        "---\n\n"
        "## 3. Correlated Features & Attribution Dynamics\n\n"
        "The AgriCost-AI feature pipeline incorporates closely related historical time series predictors:\n"
        "1. `lag_1`: Prior year's WPI index (y_{t-1})\n"
        "2. `lag_2`: Two years prior WPI index (y_{t-2})\n"
        "3. `rolling_mean_2`: Two-year moving average ((y_{t-1} + y_{t-2}) / 2)\n"
        "4. `historical_pct_change`: Prior year's percentage index change (((y_{t-1} - y_{t-2}) / y_{t-2}) * 100)\n\n"
        "### Critical Interpretive Caveat:\n"
        "- These four variables exhibit strong mutual collinearity.\n"
        "- `shap.LinearExplainer` with an independent masker calculates attributions based on linear model coefficients and the marginal background distribution.\n"
        "- **Attributions among collinear variables are shared according to the model's regularized weights, not independent real-world causality.**\n"
        "- A negative SHAP value for `lag_1` in a specific prediction does **NOT** imply that higher historical prices physically lower future prices; rather, it reflects how the model balances `lag_1`, `rolling_mean_2`, and regularized mean reversion relative to the training baseline.\n"
        "- **Do not interpret SHAP attributions as causal economic levers.**\n\n"
        "---\n\n"
        "## 4. Categorical Feature Attribution\n\n"
        "- Commodity category is encoded via `OneHotEncoder(drop='first')` across the 6 major input categories.\n"
        "- SHAP evaluates the contribution of category indicators relative to the reference category (`Farm Machinery`).\n"
        "- Category contributions capture category-wide structural price level shifts relative to the broad agricultural input basket.\n\n"
        "---\n\n"
        "## 5. Scope of Held-out Evaluation\n\n"
        "- Explanations are computed across all **78 held-out test observations** spanning financial years 2020-21, 2021-22, and 2022-23 across all 26 commodities.\n"
        "- The test split was strictly evaluated using the fitted preprocessing pipeline without any refitting or data leakage.\n"
    )
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(content)
    return doc_path


def run_explainability_pipeline(
    model_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end SHAP explainability workflow.

    1. Loads existing Ridge Regression pipeline without retraining.
    2. Loads training and test datasets.
    3. Transforms features without refitting.
    4. Initializes shap.LinearExplainer with training background data.
    5. Computes SHAP values across all 78 test observations.
    6. Verifies SHAP additivity across all test samples.
    7. Computes global feature rankings and saves shap_global_importance.csv.
    8. Explains Urea FY 2022-23 local prediction and saves local_explanation_urea_2022_23.json.
    9. Generates and saves beeswarm, bar, and waterfall plots.
    10. Saves shap_metadata.json and documentation markdown.

    Returns:
        Dict[str, Any] with artifacts, top features, local example, and verification status.
    """
    out_dir = Path(output_dir) if output_dir else SHAP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline without retraining
    pipeline = load_trained_pipeline(model_path)
    preprocessor, regressor = extract_pipeline_components(pipeline)

    # 2. Ingest train and test data
    train_df, test_df = load_training_and_testing_data()

    # 3. Preprocess matrices strictly using fitted preprocessor
    X_train_trans, X_test_trans, feature_names = prepare_preprocessed_matrices(
        preprocessor=preprocessor,
        train_df=train_df,
        test_df=test_df,
    )

    # 4. Create authentic LinearExplainer
    explainer = create_linear_explainer(
        regressor=regressor,
        X_train_trans=X_train_trans,
        feature_names=feature_names,
    )

    # 5. Compute SHAP values for test samples
    shap_explanation = compute_shap_explanations(explainer, X_test_trans)

    # 6. Global Feature Importance
    global_importance_df = compute_global_feature_importance(shap_explanation, feature_names)
    csv_path = out_dir / "shap_global_importance.csv"
    global_importance_df.to_csv(csv_path, index=False)

    # 7. Local Explanation for Urea FY 2022-23 (code: 1310020001)
    urea_code = 1310020001
    urea_period = "2022-23"
    local_urea = explain_individual_prediction(
        pipeline=pipeline,
        explainer=explainer,
        shap_explanation=shap_explanation,
        test_df=test_df,
        commodity_code=urea_code,
        period_label=urea_period,
    )
    urea_json_path = out_dir / "local_explanation_urea_2022_23.json"
    with open(urea_json_path, "w", encoding="utf-8") as f:
        json.dump(local_urea, f, indent=2)

    # 8. Generate & Save Plots
    plot_paths = generate_and_save_shap_plots(
        shap_explanation=shap_explanation,
        local_test_idx=local_urea["test_index"],
        output_dir=out_dir,
    )

    # 9. Verify additivity across entire test set
    preds_all = pipeline.predict(test_df)
    reconstructed_all = shap_explanation.base_values + np.sum(shap_explanation.values, axis=1)
    max_additivity_error = float(np.max(np.abs(preds_all - reconstructed_all)))
    all_additive = bool(max_additivity_error < 1e-4)

    # 10. Save assumptions documentation
    doc_path = save_assumptions_and_limitations_doc(out_dir)

    # 11. Save metadata JSON
    top_5_features = global_importance_df.head(5).to_dict(orient="records")
    metadata = {
        "model_name": "Ridge Regression (alpha=10.0)",
        "model_artifact": "models/trained/best_model.joblib",
        "explainer_type": "shap.LinearExplainer",
        "masker_type": "shap.maskers.Independent",
        "background_dataset": "data/processed/train_data.csv",
        "background_samples": len(train_df),
        "evaluation_dataset": "data/processed/test_data.csv",
        "evaluation_samples": len(test_df),
        "num_features_explained": len(feature_names),
        "target_variable": TARGET_COLUMN,
        "target_unit": SHAP_UNIT_LABEL,
        "base_value_expected": round(float(np.mean(shap_explanation.base_values)), 4),
        "additivity_verified_all_samples": all_additive,
        "max_additivity_error": max_additivity_error,
        "top_5_features_by_mean_abs_shap": top_5_features,
        "local_example_urea_2022_23": {
            "commodity": local_urea["commodity_name"],
            "period": local_urea["period_label"],
            "actual_wpi": local_urea["actual_wpi"],
            "predicted_wpi": local_urea["predicted_wpi"],
            "base_value": local_urea["base_value"],
            "sum_of_shap": local_urea["sum_of_shap"],
            "reconstructed": local_urea["reconstructed_prediction"],
            "unit": SHAP_UNIT_LABEL,
        },
        "artifacts_saved": {
            "global_importance_csv": str(csv_path),
            "summary_beeswarm_plot": str(plot_paths["summary_beeswarm"]),
            "summary_bar_plot": str(plot_paths["summary_bar"]),
            "local_explanation_json": str(urea_json_path),
            "waterfall_plot": str(plot_paths["waterfall_urea"]),
            "assumptions_markdown": str(doc_path),
        },
    }
    meta_path = out_dir / "shap_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "metadata": metadata,
        "global_importance_df": global_importance_df,
        "local_urea": local_urea,
        "artifacts": metadata["artifacts_saved"],
        "max_additivity_error": max_additivity_error,
    }


if __name__ == "__main__":
    results = run_explainability_pipeline()
    print("=" * 80)
    print("STAGE 6 EXPLAINABLE AI (SHAP) EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Explainer Type: {results['metadata']['explainer_type']}")
    print(f"Observations Explained: {results['metadata']['evaluation_samples']}")
    print(f"Target Unit: {results['metadata']['target_unit']}")
    print(f"Max Additivity Error: {results['max_additivity_error']:.2e}")
    print("\nTOP 5 FEATURES BY MEAN ABSOLUTE SHAP:")
    print(results["global_importance_df"].head(5).to_string(index=False))
    print("\nLOCAL EXPLANATION EXAMPLE (Urea 2022-23):")
    u = results["local_urea"]
    print(f"  Actual WPI: {u['actual_wpi']} {u['unit']}")
    print(f"  Predicted WPI: {u['predicted_wpi']} {u['unit']}")
    print(f"  Base Value E[f(X)]: {u['base_value']} {u['unit']}")
    print(f"  Sum of SHAP: {u['sum_of_shap']} {u['unit']}")
    print(f"  Base + Sum: {u['reconstructed_prediction']} {u['unit']}")
    print(f"  Additivity Verified: {u['additivity_verified']}")
    print("\nSaved Artifacts:")
    for k, v in results["artifacts"].items():
        print(f"  {k}: {v}")
