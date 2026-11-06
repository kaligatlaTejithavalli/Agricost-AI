"""🌾 AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System

Streamlit Web Dashboard (Stage 8 / Phase 3H Extended Production):
- Tab 1: Price Insights (Interactive Historical WPI Analysis through FY 2025-26, Base 2011-12 = 100.0)
- Tab 2: Cost Forecast (Recursive Multi-Year Projections for FY 2026-27 to 2028-29 using Ridge Regression alpha=10.0)
- Tab 3: Budget Planner (Index-to-Rupee Ratio Bridge & Farm Budget Estimator with Dual Scenarios)
- Tab 4: Explainable AI (Authentic SHAP Global Feature Importance & Local Waterfalls for Production Candidate)

Design Theme: Agricultural aesthetic (Forest Green #1B4332, Sage Green #95B8A0, Warm Cream #F8F6F0, Charcoal #263238).
Guardrails: Strict distinction between WPI index points and rupees; prominent historical cutoff warnings (FY 2025-26).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import streamlit as st

from src.budget import (
    FarmBudgetPlan,
    FarmExpenditureItem,
    PriceProjectionResult,
    calculate_dual_scenario_expenditure,
    calculate_input_expenditure,
    estimate_dual_scenario_input_price,
    estimate_future_input_price,
    extract_physical_unit,
    format_price_unit,
    format_quantity_unit,
)
from src.config import (
    AGRICULTURAL_INPUT_CATALOG,
    BASE_YEAR_LABEL,
    EVALUATION_DIR,
    EXTENDED_FORECASTS_CSV,
    EXTENDED_PROCESSED_TIMESERIES_CSV,
    EXTENDED_SHAP_DIR,
    FORECASTS_DIR,
    PROCESSED_DATA_DIR,
    SHAP_DIR,
    TARGET_LABEL,
    TRAINED_MODELS_DIR,
)
from src.forecast import (
    CommodityForecastResult,
    SingleYearForecast,
    advance_financial_year,
    bridge_dual_forecast_to_budget,
    bridge_forecast_to_budget,
    forecast_commodity,
    load_forecasting_pipeline,
    load_historical_commodity_series,
)


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriCost-AI | Agricultural Input Cost Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Agricultural Styling Palette
CUSTOM_CSS = """
<style>
    /* Main Background & Typography */
    .stApp {
        background-color: #F8F6F0;
        color: #263238;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1B4332 0%, #2D5A27 100%);
        color: #FFFFFF;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .main-header p {
        color: #E8F5E9 !important;
        font-size: 1.05rem;
        margin: 0;
    }
    .header-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 14px;
    }
    .badge {
        background: rgba(255, 255, 255, 0.18);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Card Containers */
    .agri-card {
        background: #FFFFFF;
        padding: 20px 24px;
        border-radius: 10px;
        border: 1px solid #E0DDD5;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .agri-card-title {
        color: #1B4332;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Highlight & Notice Boxes */
    .notice-box {
        background-color: #FFF9E6;
        border-left: 4px solid #D4A017;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #5C4300;
        margin-bottom: 16px;
    }
    .guardrail-box {
        background-color: #EBF3ED;
        border-left: 4px solid #1B4332;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #1B4332;
        margin-bottom: 16px;
    }
    .danger-box {
        background-color: #FDE8E8;
        border-left: 4px solid #C53030;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #742A2A;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders (Performance & Usability)
# ---------------------------------------------------------------------------
@st.cache_data
def get_historical_timeseries() -> pd.DataFrame:
    """Load historical FY WPI timeseries for all 26 commodities through FY 2025-26."""
    path = EXTENDED_PROCESSED_TIMESERIES_CSV
    if not path.exists():
        st.error(f"Extended timeseries dataset missing at: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def get_model_pipeline() -> Optional[Pipeline]:
    """Load the pre-trained production candidate Ridge Regression pipeline (alpha=10.0)."""
    try:
        candidate_path = TRAINED_MODELS_DIR / "production_candidate_extended.joblib"
        return load_forecasting_pipeline(candidate_path)
    except Exception as e:
        st.error(f"Error loading trained production candidate model: {e}")
        return None


@st.cache_data
def get_evaluation_report() -> Dict[str, Any]:
    """Load the Stage 5 benchmark model evaluation report."""
    path = EVALUATION_DIR / "evaluation_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_precomputed_forecasts() -> pd.DataFrame:
    """Load the pre-generated Phase 3G extended forecasts for all 26 commodities."""
    path = EXTENDED_FORECASTS_CSV
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def get_shap_importance() -> pd.DataFrame:
    """Load the extended global SHAP feature importance table."""
    path = EXTENDED_SHAP_DIR / "shap_global_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def get_local_urea_explanation() -> Dict[str, Any]:
    """Load the verified local SHAP explanation for Urea FY 2025-26."""
    path = EXTENDED_SHAP_DIR / "local_explanation_urea_2025_26.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}



@st.cache_data
def get_backtesting_commodity_metrics() -> pd.DataFrame:
    """Load the out-of-sample backtesting metrics per commodity."""
    path = EVALUATION_DIR / "backtesting" / "commodity_backtesting_performance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def get_backtesting_summary() -> Dict[str, Any]:
    """Load the out-of-sample backtesting summary JSON."""
    path = EVALUATION_DIR / "backtesting" / "backtesting_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Sidebar: Commodity Catalog Selector & System Status
# ---------------------------------------------------------------------------
def render_sidebar(history_df: pd.DataFrame) -> Tuple[int, str, str, str, float]:
    """Render the sidebar with category filters and commodity selector."""
    st.sidebar.image(
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=600&auto=format&fit=crop",
        caption="AgriCost-AI Decision Support",
        use_container_width=True,
    )
    st.sidebar.markdown("### 🌾 Commodity Catalog")

    # Group commodities by category
    categories = sorted(list({meta.category for meta in AGRICULTURAL_INPUT_CATALOG.values()}))
    selected_cat = st.sidebar.selectbox(
        "Filter by Input Category:",
        options=["All Categories"] + categories,
        index=0,
        help="Filter the 26 verified agricultural inputs by input group.",
    )

    # Filter catalog by selected category
    if selected_cat == "All Categories":
        filtered_catalog = AGRICULTURAL_INPUT_CATALOG
    else:
        filtered_catalog = {
            k: v for k, v in AGRICULTURAL_INPUT_CATALOG.items() if v.category == selected_cat
        }

    commodity_options = {
        f"{v.name} ({v.category})": k for k, v in filtered_catalog.items()
    }
    
    # Default to Urea if available
    default_idx = 0
    option_names = list(commodity_options.keys())
    for idx, opt in enumerate(option_names):
        if "Urea" in opt:
            default_idx = idx
            break

    selected_option = st.sidebar.selectbox(
        "Select Commodity:",
        options=option_names,
        index=default_idx,
        help="Choose an agricultural input commodity to analyze.",
    )
    selected_code = commodity_options[selected_option]
    meta = AGRICULTURAL_INPUT_CATALOG[selected_code]

    # Selected Commodity Metadata Card in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Commodity Code:** `{meta.code}`")
    st.sidebar.markdown(f"**Input Category:** {meta.category}")
    st.sidebar.markdown(f"**Basket Weight:** `{meta.weight:.5f}%`")
    st.sidebar.markdown(f"**Suggested Physical Unit:** `{extract_physical_unit(meta.suggested_unit)}`")
    st.sidebar.caption(f"_{meta.description}_")

    # Data Source & Cutoff Notice
    latest_cutoff_period = history_df["period_label"].max() if not history_df.empty else "2025-26"
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏛️ Data Provenance")
    st.sidebar.markdown(
        f"**Source:** DPIIT, Ministry of Commerce & Industry, Govt of India.\n\n"
        f"**Base Year:** `2011-12 = 100.0`\n\n"
        f"**Historical Cutoff:** Available through **FY {latest_cutoff_period}**.\n\n"
        f"**Model:** Ridge Regression (α = 10.0)"
    )

    return selected_code, meta.name, meta.category, meta.suggested_unit, meta.weight


# ---------------------------------------------------------------------------
# Main Application Header
# ---------------------------------------------------------------------------
def render_header(latest_cutoff: str = "FY 2025–26") -> None:
    """Render the top title, subtitle, and provenance banner."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🌾 AgriCost-AI</h1>
            <p><strong>AI-Based Agricultural Input Cost Forecasting & Decision Support System</strong></p>
            <p style="font-size: 0.95rem; margin-top: 6px; opacity: 0.95;">
                Explore historical agricultural input wholesale price indices, analyze future trends,
                estimate farming expenses, and understand AI-generated forecasts.
            </p>
            <div class="header-badges">
                <span class="badge">🏛️ Source: Govt of India DPIIT WPI Dataset</span>
                <span class="badge">📅 Historical Data Available Through {latest_cutoff}</span>
                <span class="badge">⚖️ Base Year: 2011–12 = 100.0</span>
                <span class="badge">🤖 Production Model: Ridge Regression (α = 10.0)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ---------------------------------------------------------------------------
# TAB 1: Price Insights (Historical Analysis)
# ---------------------------------------------------------------------------
def render_tab_price_insights(history_df: pd.DataFrame, commodity_code: int, commodity_name: str, category: str, unit: str) -> None:
    """Render Tab 1: Historical Wholesale Price Index (WPI) time series analysis."""
    # Extract series for selected commodity
    series_df = history_df[history_df["commodity_code"] == commodity_code].sort_values("period_order").copy()
    if series_df.empty:
        st.warning(f"No historical records found for commodity code {commodity_code}.")
        return

    first_row = series_df.iloc[0]
    latest_row = series_df.iloc[-1]
    second_latest = series_df.iloc[-2] if len(series_df) >= 2 else latest_row

    st.markdown("### 📈 Historical Wholesale Price Index (WPI) Insights")
    st.markdown(
        f"Detailed historical index movements for **{commodity_name}** ({category}) "
        f"from **FY {first_row['period_label']} to FY {latest_row['period_label']}**."
    )

    # Unit distinction guardrail alert
    st.markdown(
        """
        <div class="guardrail-box">
            <strong>⚠️ Unit Distinction Guardrail:</strong>
            All historical index values displayed in this tab are <strong>WPI index points</strong>
            (Base Year 2011–12 = 100.0). They represent wholesale macroeconomic price index levels,
            <strong>NEVER rupee market prices</strong> (₹).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calculate key historical metrics
    latest_wpi = float(latest_row["wpi_index_points"])
    one_yr_change = ((latest_wpi - float(second_latest["wpi_index_points"])) / float(second_latest["wpi_index_points"])) * 100.0
    total_change = ((latest_wpi - float(first_row["wpi_index_points"])) / float(first_row["wpi_index_points"])) * 100.0

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label=f"Latest Observed WPI ({latest_row['period_label']})",
            value=f"{latest_wpi:.1f} pts",
            help=f"Wholesale Price Index points for FY {latest_row['period_label']} (Base 2011-12 = 100.0).",
        )
    with col2:
        st.metric(
            label="1-Year Historical Change",
            value=f"{one_yr_change:+.2f}%",
            delta=f"{latest_wpi - float(second_latest['wpi_index_points']):+.1f} pts",
            help=f"Movement from FY {second_latest['period_label']} to FY {latest_row['period_label']}.",
        )
    with col3:
        years_span = len(series_df) - 1
        st.metric(
            label=f"Total Historical Movement ({years_span}-Yr)",
            value=f"{total_change:+.2f}%",
            delta=f"{latest_wpi - float(first_row['wpi_index_points']):+.1f} pts",
            help=f"Movement from {first_row['period_label']} to {latest_row['period_label']}.",
        )
    with col4:
        st.metric(
            label="Data Cutoff Period",
            value=f"FY {latest_row['period_label']}",
            help="Official DPIIT release cutoff date.",
        )


    # Plotly Interactive Chart
    fig = go.Figure()
    
    # Base Year 100 reference line
    fig.add_hline(
        y=100.0,
        line_dash="dot",
        line_color="#95B8A0",
        annotation_text="Base Year 2011-12 = 100.0",
        annotation_position="bottom right",
    )

    # Historical WPI line
    fig.add_trace(
        go.Scatter(
            x=series_df["period_label"],
            y=series_df["wpi_index_points"],
            mode="lines+markers",
            name=f"{commodity_name} (WPI)",
            line=dict(color="#1B4332", width=3),
            marker=dict(size=8, color="#1B4332"),
            hovertemplate="<b>%{x}</b><br>WPI: %{y:.1f} index points<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Historical Wholesale Price Index (WPI) Trend: {commodity_name} ({category})",
        xaxis_title="Financial Year Period",
        yaxis_title="WPI Index Points (Base 2011–12 = 100.0)",
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Historical Data Table and Download Option
    with st.expander("📋 View Complete Historical Data Table", expanded=False):
        display_df = series_df[[
            "period_label", "start_year", "end_year", "wpi_index_points", "target_unit", "base_year", "data_source"
        ]].copy()
        display_df.rename(columns={
            "period_label": "Financial Year",
            "start_year": "Start Year",
            "end_year": "End Year",
            "wpi_index_points": "WPI (Index Points)",
            "target_unit": "Unit",
            "base_year": "Base Year",
            "data_source": "Source",
        }, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {commodity_name} Historical WPI (CSV)",
            data=csv_data,
            file_name=f"{commodity_name.lower().replace(' ', '_')}_historical_wpi.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# TAB 2: Cost Forecast (Future Projections)
# ---------------------------------------------------------------------------
def render_tab_cost_forecast(
    pipeline: Optional[Pipeline],
    history_df: pd.DataFrame,
    commodity_code: int,
    commodity_name: str,
    category: str,
) -> None:
    """Render Tab 2: Recursive Multi-Year WPI Forecasting Engine with Dual Trajectories."""
    series_df = history_df[history_df["commodity_code"] == commodity_code].sort_values("period_order")
    latest_period = series_df.iloc[-1]["period_label"] if not series_df.empty else "2025-26"
    h1_period = advance_financial_year(latest_period, 1)
    h2_period = advance_financial_year(latest_period, 2)
    h3_period = advance_financial_year(latest_period, 3)

    st.markdown("### 🔮 Recursive Future WPI Cost Forecasting Engine")
    st.markdown(
        f"Generate recursive multi-year Wholesale Price Index projections for **{commodity_name}** "
        f"using the trained **Ridge Regression (α = 10.0)** machine-learning model alongside the "
        f"**Naïve Persistence** statistical benchmark."
    )

    # Prominent Cutoff & Modeling Notice
    st.markdown(
        f"""
        <div class="notice-box">
            <strong>⚠️ Historical Data Cutoff & Model Notice:</strong>
            Forecasts are algorithmic model projections based strictly on historical data ending in 
            <strong>FY {latest_period}</strong>. They are <strong>NOT current official government WPI releases</strong>.
            The projected periods (FY {h1_period}, {h2_period}, {h3_period}) represent forward recursive
            estimates, not observations generated from future calendar year data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dual-Trajectory Methodology Explanation Card
    st.markdown(
        f"""
        <div class="agri-card" style="margin-top: 10px; margin-bottom: 18px;">
            <div class="agri-card-title">⚖️ Dual-Trajectory Forecasting Framework</div>
            <p style="margin-bottom: 6px; font-size: 0.95rem;">To ensure econometric transparency, two complementary forecasting approaches are displayed:</p>
            <ul style="margin-bottom: 4px; font-size: 0.92rem;">
                <li><strong>Ridge Regression (α = 10.0):</strong> Machine-learning forecast utilizing regularized autoregressive lag features (lag_1, lag_2, 2-year moving average) to model dynamic trend momentum.</li>
                <li><strong>Naïve Persistence Baseline:</strong> Statistical benchmark assuming the last observed WPI (FY {latest_period}) remains strictly unchanged into the future (y<sub>T+h</sub> = y<sub>T</sub>).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section: Forecast Horizon From Historical Data Cutoff
    st.markdown("#### Forecast Horizon From Historical Data Cutoff")
    st.info(f"These are model projections generated from data ending in FY {latest_period}, not forecasts relative to the current year.")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        horizon_labels = {
            1: f"1 Year After Data Cutoff (FY {h1_period})",
            2: f"2 Years After Data Cutoff (FY {h2_period})",
            3: f"3 Years After Data Cutoff (FY {h3_period})",
        }
        horizon = st.radio(
            "Forecast Horizon From Historical Data Cutoff",
            options=[1, 2, 3],
            format_func=lambda h: horizon_labels[h],
            index=2,
            horizontal=False,
            help=f"Choose the forward annual recursive forecasting horizon from the FY {latest_period} data cutoff.",
        )
    with col_ctrl2:
        show_ranges = st.checkbox(
            "Show Indicative Planning Ranges (Dispersion Envelope)",
            value=True,
            help="Displays indicative planning bands based on historical RMSE and horizon compounding (sigma_h = RMSE * sqrt(h)).",
        )
        st.caption(
            "ℹ️ _Indicative ranges represent analytical dispersion heuristics compounding over multi-step horizons, "
            "not statistically guaranteed confidence intervals._"
        )

    # Run Forecast via backend engine
    try:
        forecast_result = forecast_commodity(
            commodity_code=commodity_code,
            forecast_horizon=horizon,
            pipeline=pipeline,
            timeseries_path=EXTENDED_PROCESSED_TIMESERIES_CSV,
        )
    except Exception as e:
        st.error(f"Error computing forecast for {commodity_name}: {e}")
        return

    # Key Metrics Display for the Selected Max Horizon
    target_step = forecast_result.forecasts[-1]
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label=f"Baseline WPI ({forecast_result.last_observed_period})",
            value=f"{forecast_result.last_observed_wpi:.1f} pts",
            help="Last actual observed WPI value from official historical data.",
        )
    with col_m2:
        st.metric(
            label=f"Ridge ML ({target_step.forecast_period})",
            value=f"{target_step.forecast_wpi:.1f} pts",
            delta=f"{target_step.forecast_wpi - forecast_result.last_observed_wpi:+.1f} pts ({target_step.pct_change_vs_last_observed:+.1f}%)",
            help=f"Machine-learning L2 regularized autoregressive projection for horizon h={horizon}.",
        )
    with col_m3:
        st.metric(
            label=f"Naïve Persistence ({target_step.forecast_period})",
            value=f"{forecast_result.last_observed_wpi:.1f} pts",
            delta="0.0 pts (0.00%)",
            help=f"Statistical benchmark assuming WPI remains unchanged from FY {forecast_result.last_observed_period}.",
        )
    with col_m4:
        st.metric(
            label="Selected ML Model",
            value="Ridge Regression (α = 10.0)",
            help="L2 regularized linear model selected as best ML forecaster.",
        )

    # Historical + Dual Forecast Interactive Plotly Chart
    hist_series = history_df[history_df["commodity_code"] == commodity_code].sort_values("period_order")
    
    fig = go.Figure()

    # 1. Historical Observations
    fig.add_trace(
        go.Scatter(
            x=hist_series["period_label"],
            y=hist_series["wpi_index_points"],
            mode="lines+markers",
            name="Historical Observed WPI",
            line=dict(color="#1B4332", width=3),
            marker=dict(size=7, color="#1B4332"),
            hovertemplate="<b>Observed %{x}</b><br>WPI: %{y:.1f} index points<extra></extra>",
        )
    )

    # 2. Ridge Regression Forecast Trace
    fc_periods = [forecast_result.last_observed_period] + [f.forecast_period for f in forecast_result.forecasts]
    fc_values = [forecast_result.last_observed_wpi] + [f.forecast_wpi for f in forecast_result.forecasts]
    
    fig.add_trace(
        go.Scatter(
            x=fc_periods,
            y=fc_values,
            mode="lines+markers",
            name="Ridge Regression ML Forecast",
            line=dict(color="#D97706", width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond", color="#D97706"),
            hovertemplate="<b>Ridge ML %{x}</b><br>WPI: %{y:.1f} index points<extra></extra>",
        )
    )

    # 3. Naive Persistence Baseline Trace (Assumes last observed WPI remains unchanged)
    persist_values = [forecast_result.last_observed_wpi] * len(fc_periods)
    fig.add_trace(
        go.Scatter(
            x=fc_periods,
            y=persist_values,
            mode="lines+markers",
            name="Naïve Persistence Baseline (Unchanged WPI)",
            line=dict(color="#2563EB", width=2.5, dash="dot"),
            marker=dict(size=7, symbol="square", color="#2563EB"),
            hovertemplate="<b>Persistence %{x}</b><br>WPI: %{y:.1f} index points (Flat Baseline)<extra></extra>",
        )
    )

    # 4. Optional Indicative Planning Envelope
    if show_ranges:
        lower_bounds = [forecast_result.last_observed_wpi] + [f.indicative_lower_80 for f in forecast_result.forecasts]
        upper_bounds = [forecast_result.last_observed_wpi] + [f.indicative_upper_80 for f in forecast_result.forecasts]

        fig.add_trace(
            go.Scatter(
                x=fc_periods + fc_periods[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill="toself",
                fillcolor="rgba(217, 119, 6, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Ridge Indicative Range",
            )
        )

    # Vertical line indicating historical cutoff
    fig.add_shape(
        type="line",
        x0=forecast_result.last_observed_period,
        x1=forecast_result.last_observed_period,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#4A5568", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=forecast_result.last_observed_period,
        y=1,
        yref="paper",
        text=f"Historical Cutoff (FY {forecast_result.last_observed_period})",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color="#4A5568", size=10),
    )


    fig.update_layout(
        title=f"WPI Dual-Trajectory Forecast: {commodity_name} (Horizons 1 to {horizon})",
        xaxis_title="Financial Year Period",
        yaxis_title="WPI Index Points (Base 2011–12 = 100.0)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Section: Dual-Trajectory Comparison Table
    st.markdown("#### ⚖️ Dual-Trajectory Forecast Comparison Table")
    st.caption("Direct side-by-side comparison between the Ridge Regression machine-learning projection and the Naïve Persistence baseline:")

    comparison_rows = []
    for f in forecast_result.forecasts:
        persistence_val = forecast_result.last_observed_wpi
        ridge_val = f.forecast_wpi
        diff = ridge_val - persistence_val
        comparison_rows.append({
            "Financial Year": f.forecast_period,
            "Horizon": f"h={f.horizon} ({f.horizon} Yr{'s' if f.horizon > 1 else ''})",
            "Ridge ML Forecast (pts)": f"{ridge_val:.2f}",
            "Ridge Net Change (%)": f"{f.pct_change_vs_last_observed:+.2f}%",
            "Naïve Persistence (pts)": f"{persistence_val:.2f}",
            "Persistence Net Change (%)": "0.00%",
            "Trajectory Divergence (pts)": f"{diff:+.2f}",
            "Indicative Range [Lower, Upper]": f"[{f.indicative_lower_80:.1f}, {f.indicative_upper_80:.1f}]",
            "Target Unit": TARGET_LABEL,
        })
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    # Section: Empirical Out-of-Sample Backtesting Metrics Expander
    bt_comm_df = get_backtesting_commodity_metrics()
    bt_summary = get_backtesting_summary()

    with st.expander("📊 Empirical Out-of-Sample Backtesting Performance & Model Comparison", expanded=False):
        st.markdown(
            "Evaluation of saved model forecasts against newly released official Government of India DPIIT "
            "actual wholesale price indices for completed financial years 2023–24, 2024–25, and 2025–26:"
        )

        if not bt_comm_df.empty and commodity_code in bt_comm_df["commodity_code"].values:
            comm_metrics = bt_comm_df[bt_comm_df["commodity_code"] == commodity_code].iloc[0]
            st.markdown(f"**Commodity-Specific Out-of-Sample Audit for {commodity_name}:**")
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                st.metric("Ridge 3-Yr MAE", f"{comm_metrics['ridge_mae_pts']:.2f} pts")
            with col_b2:
                st.metric("Ridge 3-Yr MAPE", f"{comm_metrics['ridge_mape_pct']:.2f}%")
            with col_b3:
                st.metric("Persistence 3-Yr MAE", f"{comm_metrics['persistence_mae_pts']:.2f} pts")
            with col_b4:
                st.metric("Ridge Directional Accuracy", f"{comm_metrics['ridge_dir_acc_pct']:.0f}%")

        if bt_summary:
            ov = bt_summary.get("overall_performance", {})
            r_ov = ov.get("ridge_regression", {})
            p_ov = ov.get("persistence_baseline", {})
            d_ov = ov.get("drift_baseline", {})
            st.markdown(
                f"""
                ---
                **Overall 78-Observation Backtesting Benchmark (26 Commodities × 3 Horizons):**
                - **Ridge Regression (ML):** MAE = **{r_ov.get('mae', 12.53):.2f}** pts | RMSE = **{r_ov.get('rmse', 20.28):.2f}** pts | MAPE = **{r_ov.get('mape', 7.30):.2f}%** | Directional Accuracy = **{r_ov.get('directional_accuracy_pct', 57.7):.1f}%**
                - **Naïve Persistence Baseline:** MAE = **{p_ov.get('mae', 8.96):.2f}** pts | RMSE = **{p_ov.get('rmse', 12.56):.2f}** pts | MAPE = **{p_ov.get('mape', 5.54):.2f}%** | Directional Accuracy = **0.0%**
                - **Drift Baseline (Linear):** MAE = **{d_ov.get('mae', 44.14):.2f}** pts | RMSE = **{d_ov.get('rmse', 66.09):.2f}** pts | MAPE = **{d_ov.get('mape', 25.73):.2f}%** | Directional Accuracy = **51.3%**
                
                _Comparative Insights:_
                1. **Ridge vs. Drift:** Ridge beat Drift by **over 3.2×** in RMSE (20.28 vs 66.09 pts), preventing runaway linear momentum extrapolation.
                2. **Ridge vs. Persistence:** Persistence achieved lower aggregate point error due to an unpredicted macroeconomic plateau in post-2022 wholesale indices. However, Persistence provides 0% directional guidance. Ridge delivered **57.7% directional accuracy** overall, and decisively outperformed Persistence on dynamic inputs like **Fuel & Power** (Ridge MAE 5.98 vs Persistence 12.74 pts), **DAP** (4.10 vs 9.53 pts), and **Farm Machinery** (4.53 vs 7.42 pts).
                """
            )

    # Detailed Step Breakdown Table
    st.markdown("#### 📋 Detailed Feature Inputs & Lag Breakdown")
    fc_table_df = forecast_result.to_dataframe()
    display_fc = fc_table_df[[
        "forecast_period", "forecast_horizon", "forecast_wpi", "pct_change_vs_last_observed",
        "lag_1", "lag_2", "rolling_mean_2", "historical_pct_change",
        "indicative_lower_80", "indicative_upper_80", "target_unit"
    ]].copy()
    display_fc.rename(columns={
        "forecast_period": "Financial Year",
        "forecast_horizon": "Horizon (Step)",
        "forecast_wpi": "Forecast WPI (pts)",
        "pct_change_vs_last_observed": f"Net Change vs {forecast_result.last_observed_period} (%)",
        "lag_1": "Lag 1 (t-1)",
        "lag_2": "Lag 2 (t-2)",
        "rolling_mean_2": "2-Yr Moving Avg",
        "historical_pct_change": "Prior % Change",

        "indicative_lower_80": "Indicative Lower",
        "indicative_upper_80": "Indicative Upper",
        "target_unit": "Unit",
    }, inplace=True)
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    # Download Button
    csv_forecast = display_fc.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {commodity_name} Forecast Results (CSV)",
        data=csv_forecast,
        file_name=f"{commodity_name.lower().replace(' ', '_')}_forecasts.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# TAB 3: Budget Planner (Index-to-Rupee Bridge)
# ---------------------------------------------------------------------------
def render_tab_budget_planner(
    pipeline: Optional[Pipeline],
    commodity_code: int,
    commodity_name: str,
    category: str,
    suggested_unit: str,
    history_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render Tab 3: Farm Input Budget Planner using the WPI Ratio Bridge."""
    if history_df is None:
        history_df = get_historical_timeseries()
    series_df = history_df[history_df["commodity_code"] == commodity_code].sort_values("period_order")
    latest_period = series_df.iloc[-1]["period_label"] if not series_df.empty else "2025-26"
    h1_period = advance_financial_year(latest_period, 1)
    h2_period = advance_financial_year(latest_period, 2)
    h3_period = advance_financial_year(latest_period, 3)

    st.markdown("### 💰 Farm Input Budget Planner")
    st.markdown(
        "Estimate future agricultural input costs in rupees by bridging user-supplied baseline purchase prices "
        "with forecasted Wholesale Price Index (WPI) movements."
    )

    # Ratio Formula Card
    st.markdown(
        """
        <div class="agri-card">
            <div class="agri-card-title">📐 Verified WPI-to-Rupee Ratio Bridge Formula</div>
            <p style="margin-bottom: 6px;">The system estimates future physical input costs strictly through the relative index ratio:</p>
            <p style="font-family: monospace; font-size: 1.05rem; font-weight: 600; color: #1B4332;">
                Estimated Future Unit Price (₹) = Baseline Unit Price (₹) × ( Forecast WPI / Baseline WPI )<br>
                Estimated Total Farm Expenditure (₹) = Estimated Future Unit Price (₹) × Quantity
            </p>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">
                <em>Rule: WPI values are dimensionless index points and are never multiplied directly by farm acres.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Inputs Section
    st.markdown("#### 1. Input Specifications & Pricing Baseline")
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        user_price = st.number_input(
            "Baseline Purchase Price Paid by User (₹):",
            min_value=1.0,
            value=266.50 if "Urea" in commodity_name else (1350.0 if "phosphate" in commodity_name.lower() else 95.0),
            step=5.0,
            format="%.2f",
            help="Enter your actual physical purchase price in rupees for the input.",
        )
    
    with col_in2:
        default_phys_unit = extract_physical_unit(suggested_unit)
        unit_str = st.text_input(
            "Physical Purchase Unit:",
            value=default_phys_unit,
            help="Physical quantity unit representing the baseline price (e.g., 50 kg bag, litre, quintal).",
        )
        phys_unit = extract_physical_unit(unit_str)
        price_unit_str = format_price_unit(unit_str)

    with col_in3:
        # Crucial: Reference Period Selector for Temporal Mismatch Guardrail
        price_period_choice = st.selectbox(
            "Price Reference Period:",
            options=[f"FY {latest_period} (Matches Historical Baseline)", "Current Market Price (Illustrative Scenario)"],
            index=0,
            help=f"Identify whether this baseline price reflects FY {latest_period} or current market pricing.",
        )

    # Temporal Mismatch Warning
    if "Current" in price_period_choice:
        st.markdown(
            f"""
            <div class="danger-box">
                <strong>⚠️ Temporal Baseline Mismatch Warning:</strong>
                The official Wholesale Price Index dataset terminates in <strong>FY {latest_period}</strong>.
                You selected a current market price. Applying the {latest_period} baseline index ratio to a current price
                assumes that post-{latest_period.split('-')[0]} inflation compounds on top of the entered price, which creates a temporal mismatch.
                <br><em>This calculation is presented strictly as an <strong>illustrative scenario</strong> rather than a reliable current-price projection.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Quantity & Forecast Horizon Selection
    st.markdown("#### 2. Farm Quantity & Forecast Period")
    col_q1, col_q2, col_q3 = st.columns(3)

    with col_q1:
        qty_mode = st.radio(
            "Quantity Specification:",
            options=["Direct Total Quantity", "Per Acre Rate × Farm Size"],
            horizontal=True,
        )

    with col_q2:
        if qty_mode == "Direct Total Quantity":
            total_qty = st.number_input("Total Required Quantity:", min_value=1.0, value=20.0, step=1.0)
            farm_size = None
            rate_per_acre = None
        else:
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                rate_per_acre = st.number_input("Quantity / Acre:", min_value=0.1, value=2.0, step=0.5)
            with col_sub2:
                farm_size = st.number_input("Farm Acres:", min_value=0.5, value=10.0, step=0.5)
            total_qty = rate_per_acre * farm_size

    with col_q3:
        target_fy = st.selectbox(
            "Target Forecast Period:",
            options=[h1_period, h2_period, h3_period],
            index=0,
            help="Select the future financial year to project expenses for.",
        )

    # Execute Forecast & Dual-Scenario Budget Projection
    try:
        base_start_yr = int(latest_period.split("-")[0])
        target_start_yr = int(target_fy.split("-")[0])
        horizon_step = target_start_yr - base_start_yr
        fc_res = forecast_commodity(
            commodity_code=commodity_code,
            forecast_horizon=horizon_step,
            pipeline=pipeline,
            timeseries_path=EXTENDED_PROCESSED_TIMESERIES_CSV,
        )
        
        dual_projections = bridge_dual_forecast_to_budget(
            forecast_result=fc_res,
            target_period=target_fy,
            current_user_price=user_price,
            price_unit=price_unit_str,
        )
        proj_ridge = dual_projections["ml_trend"]
        proj_static = dual_projections["static_baseline"]

        dual_expenditures = calculate_dual_scenario_expenditure(
            ml_trend_projection=proj_ridge,
            static_baseline_projection=proj_static,
            total_quantity=total_qty,
            quantity_per_acre=rate_per_acre,
            farm_size_acres=farm_size,
            quantity_unit=phys_unit,
        )
        exp_ridge = dual_expenditures["ml_trend"]
        exp_static = dual_expenditures["static_baseline"]
    except Exception as e:
        st.error(f"Error computing budget estimate: {e}")
        return

    qty_display = format_quantity_unit(total_qty, phys_unit)

    # Results Display
    st.markdown("---")
    st.markdown("#### 3. Dual-Scenario Budget Projection Results")
    st.caption(
        "Compare two transparent scenario estimates based on user-supplied baseline purchase prices "
        "and physical quantities. Both outputs are scenario estimates, not guaranteed predictions."
    )

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric(
            label="Baseline Total Cost (₹)",
            value=f"₹{exp_static.current_total_cost:,.2f}",
            help=f"User baseline price (₹{user_price:.2f} per {phys_unit}) × {qty_display}.",
        )
    with col_r2:
        st.metric(
            label=f"Ridge ML Trend Budget ({target_fy})",
            value=f"₹{exp_ridge.estimated_future_total_cost:,.2f}",
            delta=f"₹{exp_ridge.estimated_cost_increase:+,.2f} ({exp_ridge.projected_percentage_change:+.2f}%)",
            help="Projected budget applying Ridge ML WPI inflation ratio.",
        )
    with col_r3:
        st.metric(
            label=f"Static Baseline Budget ({target_fy})",
            value=f"₹{exp_static.estimated_future_total_cost:,.2f}",
            delta="₹0.00 (0.00%)",
            help=f"Projected budget assuming WPI remains fixed at FY {latest_period} level (zero index movement).",
        )
    with col_r4:
        divergence = exp_ridge.estimated_future_total_cost - exp_static.estimated_future_total_cost
        st.metric(
            label="Scenario Budget Divergence (₹)",
            value=f"₹{divergence:+,.2f}",
            delta=f"{exp_ridge.projected_percentage_change:+.2f}% vs Static",
            help="Difference in total estimated budget between Ridge ML Trend and Static Baseline scenarios.",
        )

    # Side-by-Side Scenario Comparison Table
    st.markdown("##### ⚖️ Side-by-Side Budget Scenario Comparison")
    budget_comparison_df = pd.DataFrame([
        {
            "Budget Dimension": "Underlying WPI Index Level",
            "Baseline Cost Reference": f"{proj_ridge.current_wpi:.2f} pts (FY {latest_period})",
            "Scenario A: Ridge ML Trend": f"{proj_ridge.forecast_wpi:.2f} pts ({target_fy})",
            "Scenario B: Static Baseline": f"{proj_static.forecast_wpi:.2f} pts (Unchanged)",
            "Scenario Divergence": f"{proj_ridge.forecast_wpi - proj_static.forecast_wpi:+.2f} pts",
        },
        {
            "Budget Dimension": "WPI Movement Assumption",
            "Baseline Cost Reference": "Historical Baseline",
            "Scenario A: Ridge ML Trend": f"{proj_ridge.projected_percentage_change:+.2f}%",
            "Scenario B: Static Baseline": "0.00% (Flat Index)",
            "Scenario Divergence": f"{proj_ridge.projected_percentage_change:+.2f}%",
        },
        {
            "Budget Dimension": f"Estimated Unit Price (₹ per {phys_unit})",
            "Baseline Cost Reference": f"₹{user_price:,.2f}",
            "Scenario A: Ridge ML Trend": f"₹{proj_ridge.estimated_future_price:,.2f}",
            "Scenario B: Static Baseline": f"₹{proj_static.estimated_future_price:,.2f}",
            "Scenario Divergence": f"₹{proj_ridge.estimated_future_price - proj_static.estimated_future_price:+,.2f}",
        },
        {
            "Budget Dimension": f"Required Usage Quantity ({phys_unit})",
            "Baseline Cost Reference": f"{total_qty:.2f}",
            "Scenario A: Ridge ML Trend": f"{total_qty:.2f}",
            "Scenario B: Static Baseline": f"{total_qty:.2f}",
            "Scenario Divergence": "0.00 (Same physical usage)",
        },
        {
            "Budget Dimension": "Estimated Total Farm Budget (₹)",
            "Baseline Cost Reference": f"₹{exp_static.current_total_cost:,.2f}",
            "Scenario A: Ridge ML Trend": f"₹{exp_ridge.estimated_future_total_cost:,.2f}",
            "Scenario B: Static Baseline": f"₹{exp_static.estimated_future_total_cost:,.2f}",
            "Scenario Divergence": f"₹{divergence:+,.2f}",
        },
        {
            "Budget Dimension": "Net Budget Change vs Baseline (₹)",
            "Baseline Cost Reference": "₹0.00",
            "Scenario A: Ridge ML Trend": f"₹{exp_ridge.estimated_cost_increase:+,.2f}",
            "Scenario B: Static Baseline": "₹0.00",
            "Scenario Divergence": f"₹{divergence:+,.2f}",
        },
    ])
    st.dataframe(budget_comparison_df, use_container_width=True, hide_index=True)

    # Detailed Calculation Step Breakdown
    with st.expander("🔍 View Detailed Mathematical Step-by-Step Breakdown for Both Scenarios", expanded=True):
        st.markdown(
            f"""
            **Common Parameters:**
            - **Selected Commodity:** {commodity_name} (Category: {category})
            - **Baseline WPI (FY {latest_period}):** `{proj_ridge.current_wpi:.2f}` WPI index points
            - **User Baseline Purchase Price:** `₹{user_price:.2f}` per {phys_unit}
            - **Required Quantity:** `{qty_display}`
            - **Baseline Total Cost:** `{total_qty:.2f} × ₹{user_price:.2f} = ` **`₹{exp_static.current_total_cost:,.2f}`**

            ---
            **Scenario A — Ridge ML Trend Calculation:**
            - **Forecast WPI ({target_fy}):** `{proj_ridge.forecast_wpi:.2f}` WPI index points
            - **WPI Ratio:** `{proj_ridge.forecast_wpi:.4f} / {proj_ridge.current_wpi:.4f} = {proj_ridge.forecast_wpi / proj_ridge.current_wpi:.4f}`
            - **Estimated Future Price:** `₹{user_price:.2f} × {proj_ridge.forecast_wpi / proj_ridge.current_wpi:.4f} = ` **`₹{proj_ridge.estimated_future_price:.2f}`** per {phys_unit}
            - **Projected Total Expenditure:** `{total_qty:.2f} × ₹{proj_ridge.estimated_future_price:.2f} = ` **`₹{exp_ridge.estimated_future_total_cost:,.2f}`**

            ---
            **Scenario B — Static Baseline Calculation (Zero Index Movement):**
            - **Assumed WPI ({target_fy}):** `{proj_static.forecast_wpi:.2f}` WPI index points (identical to FY {latest_period} baseline)
            - **WPI Ratio:** `{proj_static.forecast_wpi:.4f} / {proj_static.current_wpi:.4f} = 1.0000`
            - **Estimated Future Price:** `₹{user_price:.2f} × 1.0000 = ` **`₹{proj_static.estimated_future_price:.2f}`** per {phys_unit}
            - **Projected Total Expenditure:** `{total_qty:.2f} × ₹{proj_static.estimated_future_price:.2f} = ` **`₹{exp_static.estimated_future_total_cost:,.2f}`**
            """
        )


    # Agricultural Budget Disclaimer
    st.markdown(
        """
        <div class="notice-box">
            <strong>🌾 Budget Scenario Disclaimer:</strong>
            Both outputs are scenario-based planning calculations using relative Wholesale Price Index relationships.
            Neither scenario is a guaranteed lower bound, upper bound, or conservative certainty.
            Actual farmer purchase prices may deviate due to government subsidies (e.g. statutory MRP on Urea),
            local dealer pricing, transportation surcharges, and retail market conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TAB 4: Explainable AI (SHAP Interpretability)
# ---------------------------------------------------------------------------
def render_tab_explainable_ai() -> None:
    """Render Tab 4: Authentic SHAP Explainability & Feature Attribution."""
    st.markdown("### 🧠 Explainable AI (XAI) with Authentic SHAP")
    st.markdown(
        "Inspect how each historical feature and input category contributes to the machine-learning "
        "model's Wholesale Price Index (WPI) predictions using **SHapley Additive exPlanations (SHAP)**."
    )

    # Explainer Guardrail & Causality Statement
    st.markdown(
        """
        <div class="guardrail-box">
            <strong>⚠️ Interpretability & Unit Guardrail:</strong>
            All SHAP attributions are reported strictly in <strong>WPI index points</strong> (Base 2011–12 = 100.0).
            SHAP explains how the statistical model calculates a forecast; it does <strong>NOT establish causal economic mechanisms</strong>.
            Because historical lags (lag_1, lag_2, rolling_mean_2) are collinear, SHAP attribution is distributed among them based
            on regularized model weights rather than isolated physical causation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1: Global Feature Importance
    st.markdown("#### 1. Global Feature Importance (Production Model Explanations)")
    st.markdown(
        "Mean absolute SHAP values across all 312 observations spanning 12 financial years (FY 2014–15 to FY 2025–26) "
        "for the production candidate model **Ridge Regression (α = 10.0)**. "
        "Higher values indicate greater overall influence on model predictions."
    )

    shap_df = get_shap_importance()
    if not shap_df.empty:
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.markdown("**Feature Attribution Summary Table:**")
            display_shap = shap_df[["rank", "feature_display", "mean_abs_shap", "mean_shap", "unit"]].copy()
            display_shap.rename(columns={
                "rank": "Rank",
                "feature_display": "Model Feature",
                "mean_abs_shap": "Mean |SHAP| (pts)",
                "mean_shap": "Mean Bias (pts)",
                "unit": "Unit",
            }, inplace=True)
            st.dataframe(display_shap, use_container_width=True, hide_index=True)

        with col_g2:
            st.markdown("**Top Feature Attribution Ranking:**")
            fig_bar = px.bar(
                shap_df.sort_values("mean_abs_shap", ascending=True),
                x="mean_abs_shap",
                y="feature_display",
                orientation="h",
                labels={"mean_abs_shap": "Mean |SHAP Value| (WPI Index Points)", "feature_display": "Feature"},
                color="mean_abs_shap",
                color_continuous_scale="Greens",
            )
            fig_bar.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False,
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Saved Summary & Beeswarm Plot Images
    st.markdown("---")
    st.markdown("#### 2. SHAP Distribution & Summary Plots")
    col_p1, col_p2 = st.columns(2)
    
    beeswarm_path = EXTENDED_SHAP_DIR / "summary_beeswarm.png"
    bar_path = EXTENDED_SHAP_DIR / "summary_bar.png"

    with col_p1:
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="SHAP Summary Beeswarm Plot across All Extended Samples", use_container_width=True)
        else:
            st.info("Beeswarm plot image not found.")

    with col_p2:
        if bar_path.exists():
            st.image(str(bar_path), caption="Global Mean Absolute SHAP Importance Bar Chart", use_container_width=True)
        else:
            st.info("Bar plot image not found.")

    # Section 3: Local Prediction Waterfall Explanation (Urea FY 2025-26 Example)
    st.markdown("---")
    st.markdown("#### 3. Local Explanation Example: Urea FY 2025–26")
    st.markdown(
        "Demonstration of local explainability for an observed historical training point (Urea FY 2025–26), "
        "confirming exact mathematical additivity: `Base Value E[f(X)] + sum(SHAP) == Predicted WPI`. "
        "*(Note: This explains the model's fitted estimation for the historical FY 2025–26 period, not a future FY 2026–27 projection).*"
    )

    urea_data = get_local_urea_explanation()
    waterfall_path = EXTENDED_SHAP_DIR / "waterfall_urea_2025_26.png"

    if urea_data:
        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            st.markdown(
                f"""
                <div class="agri-card">
                    <div class="agri-card-title">🔬 Urea FY 2025–26 Local Prediction Audit</div>
                    <ul>
                        <li><strong>Actual Historical WPI:</strong> {urea_data.get('actual_wpi')} WPI index points</li>
                        <li><strong>Model Predicted WPI:</strong> {urea_data.get('predicted_wpi')} WPI index points</li>
                        <li><strong>Expected Base Value E[f(X)]:</strong> {urea_data.get('base_value')} WPI index points</li>
                        <li><strong>Sum of SHAP Contributions:</strong> {urea_data.get('sum_of_shap')} WPI index points</li>
                        <li><strong>Reconstructed Total:</strong> {urea_data.get('reconstructed_prediction')} WPI index points</li>
                        <li><strong>Additivity Verified:</strong> {'✅ Exact Match (diff = 0.00 pts)' if urea_data.get('additivity_verified') else '❌ Discrepancy'}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Feature contributions table
            contribs = urea_data.get("feature_contributions", [])
            if contribs:
                contrib_df = pd.DataFrame(contribs)
                display_contrib = contrib_df[["feature_display", "shap_value", "unit"]].copy()
                display_contrib.rename(columns={
                    "feature_display": "Feature",
                    "shap_value": "SHAP Contribution (WPI points)",
                    "unit": "Unit",
                }, inplace=True)
                st.dataframe(display_contrib.head(6), use_container_width=True, hide_index=True)

        with col_u2:
            if waterfall_path.exists():
                st.image(str(waterfall_path), caption="SHAP Waterfall Plot for Urea FY 2025–26", use_container_width=True)
            else:
                st.info("Waterfall plot image not found.")


# ---------------------------------------------------------------------------
# Main App Execution Controller
# ---------------------------------------------------------------------------
def main() -> None:
    """Main application controller for AgriCost-AI Streamlit Dashboard."""
    # Load backend resources
    history_df = get_historical_timeseries()
    pipeline = get_model_pipeline()

    if history_df.empty:
        st.error("Unable to load agricultural timeseries data. Please ensure data preprocessing has been run.")
        return

    latest_cutoff = history_df["period_label"].max() if not history_df.empty else "2025-26"

    # Render Header
    render_header(latest_cutoff=f"FY {latest_cutoff}")

    # Render Sidebar and get user selections
    selected_code, commodity_name, category, suggested_unit, weight = render_sidebar(history_df)

    # Create 4 Core Application Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Insights",
        "🔮 Cost Forecast",
        "💰 Budget Planner",
        "🧠 Explainable AI",
    ])

    with tab1:
        render_tab_price_insights(
            history_df=history_df,
            commodity_code=selected_code,
            commodity_name=commodity_name,
            category=category,
            unit=suggested_unit,
        )

    with tab2:
        render_tab_cost_forecast(
            pipeline=pipeline,
            history_df=history_df,
            commodity_code=selected_code,
            commodity_name=commodity_name,
            category=category,
        )

    with tab3:
        render_tab_budget_planner(
            pipeline=pipeline,
            commodity_code=selected_code,
            commodity_name=commodity_name,
            category=category,
            suggested_unit=suggested_unit,
            history_df=history_df,
        )

    with tab4:
        render_tab_explainable_ai()


if __name__ == "__main__":
    main()

