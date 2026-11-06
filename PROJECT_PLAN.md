# AgriCost AI — Architectural Blueprint & Implementation Plan
**AI-Based Agricultural Input Cost Forecasting & Decision Support System**

---

## 1. Executive Summary & Academic Scope

- **Student / Project Level:** Final-Year B.Tech Capstone Project (Computer Science / Agricultural Engineering / Data Science).
- **Project Title:** AgriCost AI: AI-Based Agricultural Input Cost Forecasting.
- **Core Problem:** Agricultural input costs—fertilizers, certified seeds, agricultural labour, machinery rentals, and diesel fuel—fluctuate significantly due to macroeconomic trends, seasonality, supply chain disruptions, and government subsidy structures. Unpredictable input expenses reduce farm profitability and hinder pre-sowing financial planning for smallholder farmers and agribusinesses.
- **Proposed Solution:** A lightweight, explainable, and production-ready machine learning web application that:
  1. Ingests genuine historical agricultural input cost data with documented provenance.
  2. Applies leak-free chronological time-series feature engineering.
  3. Evaluates regression forecasting models against benchmark baselines (MAE and RMSE).
  4. Delivers transparent feature attribution using SHAP (SHapley Additive exPlanations).
  5. Provides a practical per-acre Farm Budget Planner and modern Streamlit interface.

---

## 2. Workspace Status

- **Working Directory:** `c:\Users\kteji\AgriCost-AI`
- **Initial Inspection Result:** The directory is currently clean and empty.
- **Status:** Ready for structured module initialization once this foundational plan is approved.

---

## 3. Requirements Analysis & Academic Objectives

| Component | Academic Requirement | Implementation Standard |
| :--- | :--- | :--- |
| **Data Integrity** | Genuine historical datasets only; no synthetic fabrication. | Source-verified government dataset: `INDIAN_GOVT_WHOLESALE_PRICE_INDEX.csv` (Office of the Economic Adviser, DPIIT, Base Year 2011-12 = 100.0). |
| **Target Variable** | Forecast target strictly distinct from rupee amounts. | **`WPI index points`** (dimensionless statistical index normalized to Base 2011-12 = 100.0). Never presented directly as rupees. |
| **Budget Bridge** | Transparent economic cost estimation. | User supplies current rupee price ($₹$); system estimates future price using the validated index ratio: $\text{Price}_{\text{future}} = \text{Price}_{\text{user}} \times \left( \frac{\text{WPI}_{\text{forecast}}}{\text{WPI}_{\text{current}}} \right)$. |
| **Validation Strategy** | Time-series chronological split. | No random shuffle; train on earlier historical periods (e.g. 2012-13 to 2019-20), test on subsequent held-out years (2020-21 to 2022-23). |
| **Evaluation** | Statistical index error metrics. | MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) measured in **WPI index points**; benchmarked against Naive Baseline. |
| **Explainability** | Authentic Model Interpretability (XAI). | SHAP (TreeExplainer or LinearExplainer) running live against the trained model features. |
| **User Interface** | Interactive, accessible dashboard. | Streamlit web application with 4 distinct tabs (Price Insights, Cost Forecast, Budget Planner, Explainable AI). |
| **Visual Aesthetics** | Modern agricultural theme. | Custom typography, restrained earthy palette (forest green, warm cream, neutral slate), uncluttered cards, zero unnecessary animations. |
| **Deployment** | Streamlit Community Cloud readiness. | Modular codebase, pinned `requirements.txt`, reproducible dependencies, standard headless configuration. |

---

## 4. Necessary Data Fields & Target Variable Selection

### 4.1 Target Variable Definition

To ensure academic and economic rigor:
- **Primary Forecasting Target:** **`WPI index points`** (Base Year 2011-12 = 100.0) for the selected agricultural input commodity.
  - Stored and displayed with an explicit label: **`WPI index points`**.
  - Must **never** be labeled as rupees (₹), price per bag, price per kilogram, or price per litre.
- **Economic Bridge to Farm Budget:**
  - Farmers pay input costs in rupees per physical unit (₹/50kg bag, ₹/litre, ₹/hour).
  - The application asks the user for their current baseline rupee cost and physical unit.
  - The model calculates the projected percentage change from historical to forecast WPI.
  - The future price is estimated using the verified index-ratio formula:
    $$\text{estimated\_future\_price} = \text{current\_user\_price} \times \left( \frac{\text{forecast\_wpi}}{\text{current\_wpi}} \right)$$
    $$\text{projected\_percentage\_change} = \left( \frac{\text{forecast\_wpi} - \text{current\_wpi}}{\text{current\_wpi}} \right) \times 100$$
- **Commodity Isolation (Inputs vs. Outputs):**
  - **Inputs Included:** Fertilizers (Urea, DAP, Mixed fertilizer, Superphosphate), Crop Protection (Pesticides, Insecticides, Fungicides), Fuel & Power (High Speed Diesel, Petrol, Electricity), Farm Machinery (Agricultural tractors, Implements, Harvesters, Threshers), Livestock Feeds (Fodder, Cattle feed compounds).
  - **Outputs Strictly Excluded:** Primary Food Articles / Crops (Bajra, Masur, Tomato, Cauliflower, Apple, Turmeric) and commercial oilseeds (Groundnut seed, Mustard seed) represent farm harvest revenues, NOT input costs.
  - **Labour Wages:** Labour wages are not collected in WPI and are not fabricated.

### 4.2 Data Schema & Provenance Structure

The ingestion pipeline validates that the raw government dataset adheres to the official DPIIT structure:

| Column Name | Data Type | Description | Range / Example |
| :--- | :--- | :--- | :--- |
| `COMM_NAME` | `string` | Official DPIIT commodity or group designation | `Di ammonium phosphate` |
| `COMM_CODE` | `int64` | 10-digit hierarchical classification code | `1310020002` |
| `COMM_WT` | `float64` | Weight in the national WPI basket (out of 100.0) | `0.07425` |
| `INDEX2011` to `INDEX2012` | `float64` / `NA` | Unpopulated historical columns in published source | Strictly `NA` (not imputed) |
| `INDEX20122013` to `INDEX20222023` | `float64` | Populated Financial Year (FY) series (Apr–Mar) | `38.3` to `410.3` index points |
| `INDEX2013` to `INDEX2022` | `float64` | Populated Calendar Year (CY) series (Jan–Dec) | `59.2` to `413.5` index points |

---

## 5. Potential Data Limitations & Mitigation Strategy

1. **Wholesale Index vs Retail Farmgate Pass-Through:**
   - *Limitation:* WPI tracks bulk wholesale transactions. Retail farmgate prices paid by smallholder farmers may experience dealer markups, local transportation surcharges, or supply bottlenecks.
   - *Mitigation:* The system explicitly documents this pass-through assumption and presents all future rupee prices as statistical planning estimates, not guaranteed quotes.
2. **Subsidized Price Stability (Urea):**
   - *Limitation:* Urea is subject to statutory MRP under government subsidy, showing minimal historical index variance (101.7 to 115.1 over 10 years).
   - *Mitigation:* Document the policy-driven nature of Urea; contrast with market-driven inputs (DAP, Diesel, Machinery).
3. **Temporal Incompatibility (FY vs CY):**
   - *Limitation:* The raw dataset interleaves Financial Years (Apr–Mar) and Calendar Years (Jan–Dec).
   - *Mitigation:* Strictly segregate series in `data_loader.py`. The Financial Year series (11 annual observations from 2012-13 to 2022-23) is the primary default series for agricultural planning.
4. **Usage Assumption Integrity:**
   - *Limitation:* Crop input application rates (kg/acre or bags/acre) vary widely by soil chemistry, agro-climatic zones, and irrigation.
   - *Mitigation:* The Budget Planner never invents hardcoded crop usage assumptions. It prompts the farmer for their planned application quantity per acre and farm size.

---

## 6. System Architecture & Modular Folder Structure

The project avoids monolithic single-file scripts and is structured into decoupled, testable components:

```
AgriCost-AI/
├── .streamlit/
│   └── config.toml                 # Custom palette & headless server configuration
├── data/
│   ├── raw/                        # Immutable raw historical records with provenance metadata
│   │   └── data_dictionary.md      # Data definitions, units, sources, and date spans
│   └── processed/                  # Normalized, validated, and feature-engineered datasets
├── models/
│   ├── trained/                    # Serialized models (.joblib), scalers, and feature metadata
│   └── evaluation/                 # Metrics logs (MAE, RMSE, baseline comparisons)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths, UI color tokens, feature definitions
│   ├── data_loader.py              # Data ingestion, schema validation, provenance checks
│   ├── preprocessor.py             # Quality checks, outlier audits, chronological sorting
│   ├── feature_engineering.py      # Lag variables, rolling averages, seasonal cyclical encodings
│   ├── train.py                    # Chronological split, baseline trainer, model trainer, metrics
│   ├── forecast.py                 # Multi-step future horizon projection engine
│   ├── explainability.py           # SHAP summary & force/waterfall explanation generators
│   └── budget.py                   # Per-acre farm budgeting calculator linking forecasts to costs
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py         # Schema tests, unit consistency, missing-value assertions
│   ├── test_features.py            # Chronological integrity and data leakage prevention tests
│   ├── test_model.py               # MAE/RMSE calculation tests, baseline benchmark tests
│   └── test_budget.py              # Per-acre budget calculation sanity tests
├── app.py                          # Streamlit interactive web application
├── requirements.txt                # Pinned dependencies for local and cloud deployment
├── runtime.txt                     # Target Python version for Streamlit Cloud
├── PROJECT_PLAN.md                 # This foundational architecture document
└── README.md                       # Comprehensive final project documentation & viva guide
```

---

## 7. Mapping Architecture to the 18 Mandatory Workflow Steps

| Step # | Mandatory Workflow Step | Responsible Module / File | Academic Output / Artifact |
| :---: | :--- | :--- | :--- |
| **1** | Define objective and forecast scope | `PROJECT_PLAN.md`, `src/config.py` | Scope definition: Rupee unit input price forecast for farming operations. |
| **2** | Identify agricultural input components | `src/config.py`, `data/raw/` | Component selection: Fertilizers (DAP/MOP), Field Labour, Machinery/Diesel. |
| **3** | Collect historical data from reliable sources | `data/raw/`, `data_loader.py` | Verified historical records with provenance citations and unit checks. |
| **4** | Understand seasonal and regional patterns | `src/preprocessor.py`, `app.py` | Exploration of Kharif vs. Rabi peak price pressures and regional differences. |
| **5** | Check data quality, missing values, outliers | `src/preprocessor.py`, `test_data_loader.py` | Missingness reports, distribution bounds, outlier documentation. |
| **6** | Clean and preprocess numerical data | `src/preprocessor.py` | Imputation policies (if any), chronological sorting, standard types. |
| **7** | Perform exploratory data analysis (EDA) | `app.py` (Tab 1: Price Insights) | Interactive statistical distributions, box plots, year-over-year summaries. |
| **8** | Visualize trends and seasonality | `app.py` (Tab 1), Plotly charts | Seasonal cycle decompositions, historical line trends with harvest markers. |
| **9** | Engineer time-related features | `src/feature_engineering.py` | Lag variables ($t-1, t-3, t-12$), rolling means ($3m, 6m$), sine/cosine seasonal encodings. |
| **10** | Split data into train and test sets | `src/train.py`, `test_features.py` | **Strict chronological split** (e.g., 80% past data for train, 20% recent for test; no shuffling). |
| **11** | Train regression or forecasting model | `src/train.py` | Baseline (Naive Persistence, Drift, Moving Average) vs. Trained Regressors (Ridge, Random Forest, Gradient Boosting). |
| **12** | Evaluate model using MAE and RMSE | `src/train.py`, `models/evaluation/` | MAE and RMSE evaluated in **WPI index points**; Drift and Naive baselines benchmarked against Ridge Regression. |
| **13** | Interpret forecasted trends | `src/forecast.py`, `app.py` (Tab 2) | Future horizon projections with historical context and uncertainty bounds. |
| **14** | Save trained model for deployment | `models/trained/best_model.joblib` | Serialized Ridge ML pipeline, metadata, and evaluation reports. |
| **15** | Build Streamlit interface (Year & Season) | `app.py` | 4-Tab Web App: Price Insights, Cost Forecast, Budget Planner, Explainable AI. |
| **16** | Test predictions using representative cases | `tests/`, `app.py` | Pytest suite validating input scenarios (e.g., Kharif 2025, high-inflation scenario). |
| **17** | Prepare for Streamlit Community Cloud | `requirements.txt`, `runtime.txt`, `.streamlit/config.toml` | Cloud-ready, memory-conscious, zero-breakage deployment configuration. |
| **18** | Document assumptions, limitations, conclusions | `README.md`, `app.py` | Academic report detailing model scope, limitations, and future work. |

---

## 8. Accuracy Risks & Prevention Strategies

| Threat to Academic Rigor | Mechanism of Error | Prevention Strategy in AgriCost AI |
| :--- | :--- | :--- |
| **Data Leakage** | Standard random `train_test_split` randomly distributes consecutive months across train and test sets. The model memorizes adjacent points, yielding deceptively high $R^2$ (> 0.99) that collapses on future data. | Use **strictly chronological splitting** (`TimeSeriesSplit` or fixed date cutoff). Fit scalers and feature transformers strictly on the training partition. |
| **Look-Ahead Bias** | Creating rolling averages that include the current or future timestep ($t$ or $t+1$). | Calculate rolling features strictly on lagged values: e.g., $\text{RollMean}_{3}(t) = \text{mean}(y_{t-1}, y_{t-2}, y_{t-3})$. |
| **Spurious Accuracy / Baseline Absence** | Claiming a complex model "works well" without benchmarking against simple persistence. | Establish an explicit **Naive Persistence Baseline** (predicting $y_t = y_{t-1}$) and **Seasonal Naive Baseline** ($y_t = y_{t-12}$). A model is only viable if its RMSE and MAE are statistically lower than the baseline. |
| **Unit & Concept Conflation** | Treating Wholesale Price Index (WPI) numbers as rupee costs, or mixing ₹/quintal with ₹/50kg bag. | Implement rigid schema validations in `data_loader.py` that enforce distinct units and reject mismatched joins. |
| **Fabricated Explainability** | Generating hardcoded or mock feature importance charts. | Derive SHAP values dynamically from the actual trained scikit-learn estimator using test feature matrices. |
| **Over-Promised Certainty** | Presenting future forecasts as exact facts to farmers. | Express all forward forecasts with explicit confidence/prediction bands and a prominent disclaimer stating forecasts are statistical estimates. |

---

## 9. User Interface Design System (Clean Agricultural Aesthetic)

The Streamlit UI will adhere to a clean, restrained, high-contrast visual design:
- **Primary Color Palette:**
  - Deep Forest Green: `#2D5A27` (Headers, key metrics, active states)
  - Earthy Olive Accent: `#4A7C59` (Secondary accents, subtle borders)
  - Warm Cream / Off-White: `#FBFBF9` (Main background for reduced eye strain)
  - Crisp Card White: `#FFFFFF` (Card surfaces, clean white container cards)
  - Neutral Charcoal: `#242E28` (Primary typography, high contrast)
  - Muted Sage: `#7A8B7B` (Subtitles, metadata, axis labels)
- **Typography:** Clean sans-serif hierarchy (inter-compatible system fonts: Segoe UI, Roboto, Helvetica Neue).
- **Navigation Structure (4 Core Tabs):**
  1. 🌾 **Price Insights:** Interactive time series plots, seasonal decomposition, historical inflation trends.
  2. 🔮 **Cost Forecast:** Interactive inputs (Target Year, Season/Month, Input Category), forecasted price with prediction interval.
  3. 📊 **Budget Planner:** Select crop type, farm size (acres), and estimated input usage; computes total forecasted input budget vs historical baseline.
  4. 🧠 **Explainable AI:** SHAP summary plots and individual prediction waterfall breakdowns revealing which historical lags and seasonal indicators drove the prediction.

---

## 10. Acceptance Criteria for Project Completion

1. [x] **Data Provenance:** Historical dataset documented with source, physical units, and date range in `data/raw/data_dictionary.md`.
2. [x] **Clean Preprocessing:** Zero silent imputations; unit validation enforced; chronological sorting verified.
3. [x] **Leak-Free Features:** Feature engineering verified to use only past information ($t-1$ and earlier).
4. [x] **Model Benchmark:** Regression model (Ridge / Random Forest / Gradient Boosting) trained on training split and benchmarked against Baselines on held-out test set.
5. [x] **Valid Metrics:** MAE and RMSE reported strictly in **WPI index points** with clear real-world interpretation.
6. [x] **Authentic SHAP:** SHAP analysis accurately computes feature importance from the actual trained Ridge pipeline (`LinearExplainer`).
7. [x] **Complete Streamlit Application:** All 4 tabs functional without errors or warnings (`app.py`).
8. [x] **Farm Budget Integration:** Dynamic per-acre calculation correctly utilizes the forecasted price values via ratio bridge (`budget.py`).
9. [x] **Phase 3A Historical Data Integration & Testing:** Staged DPIIT actuals (FY 2023-24 to 2025-26) verified and integrated into extended processed dataset (`data/processed/agricultural_inputs_wpi_fy_timeseries_extended.csv`, 364 rows) and leak-free feature engineering (312 rows); full test suite passes with 100% success (161 passed, 0 failed across all 11 modules).
10. [x] **Phase 3B Updated Model Training & Chronological Evaluation:** Chronological partition evaluation (Train: FY 2014-15 through FY 2022-23 [234 samples], Test: FY 2023-24 through FY 2025-26 [78 samples]) with zero leakage; baselines and ML regressors evaluated in WPI index points; versioned artifacts saved under `models/evaluation/phase_3b/`; separate production candidate trained on all 312 observations through FY 2025-26 and saved under `models/trained/production_candidate_extended.joblib` without overwriting `best_model.joblib`; full test suite passes with 100% success (175 passed, 0 failed across all 12 modules).
11. [x] **Phase 3C Recursive Multi-Year Forecast Validation:** Retrospective evaluation of multi-year recursive forecasts across 26 agricultural input commodities across Horizons 1, 2, and 3 (FY 2023-24, FY 2024-25, FY 2025-26, N=78 observations) using models cut off at FY 2022-23 (N=234 training records); baseline benchmarking (Naïve Persistence, Drift, Rolling Mean) and ML pipelines; validation report and metrics saved under `models/evaluation/phase_3c/`; 10 automated unit and integration tests passing in `tests/test_phase_3c.py`; full test suite passes with 100% success (185 passed, 0 failed across all 13 modules).
12. [x] **Phase 3D Dual-Trajectory Forecasting & Budgeting Implementation:** Implemented dual-trajectory decision support architecture across `src/forecast.py`, `src/budget.py`, and `app.py`. Integrates Ridge Regression ML trend projection alongside the Static Baseline (unchanged index carry-forward) in Tab 2 (Cost Forecast) and Tab 3 (Budget Planner). Features side-by-side scenario comparison table, divergence analysis, and step-by-step ratio bridge breakdown without retraining or replacing `best_model.joblib`. 12 automated tests in `tests/test_dual_trajectory.py`; full test suite passes with 100% success (197 passed, 0 failed across all 14 modules).
13. [ ] **Deployment Readiness & Production Cutover:** Deployment configuration files prepared (`requirements.txt`, `runtime.txt`), but application deployment, deployed model replacement, and future recursive forecasting updates remain strictly pending future authorization.

---

## 11. Staged Execution Plan

Following our stage-by-stage development process:
- **Stage 1 (Completed):** Requirements, architecture, data scope, and project plan approval.
- **Stage 2 (Completed):** Historical dataset sourcing, verification, and data dictionary documentation.
- **Stage 3 (Completed):** Data loading, schema validation, and cleaning pipeline (`data_loader.py` & `preprocessor.py`).
- **Stage 4 (Completed):** Feature engineering & chronological split pipeline (`feature_engineering.py`).
- **Stage 5 (Completed):** Model training, baseline benchmarking, and evaluation (`train.py`).
- **Stage 6 (Completed & Verified):** Explainable AI integration with authentic SHAP (`explainability.py`).
- **Stage 7 (Completed & Verified):** Future-year recursive forecasting module (`forecast.py`) and decision support.
- **Stage 8 (Completed & Verified):** Streamlit multi-tab user interface implementation (`app.py`).
- **Stage 9 Phase 1 & 2 (Completed & Verified):** Automated unit testing suite and out-of-sample backtesting against official DPIIT actuals (`backtesting.py`).
- **Phase 3A (Completed & Verified):** Historical data integration extending processed timeseries through FY 2025–26 (364 observations across 26 commodities) and leak-free feature engineering (312 observations); verified with 18 automated tests in `tests/test_phase_3a.py`.
- **Phase 3B (Completed & Verified):** Model training & chronological evaluation on extended historical dataset (`src/train_extended.py`). Evaluated Naïve Persistence, Drift, Rolling Mean, Ridge ($\alpha=1.0, 10.0$), Random Forest, and Gradient Boosting under a strict chronological split (FY 2014–15 to FY 2022–23 for training, FY 2023–24 to FY 2025–26 for testing). Verified with 14 automated tests in `tests/test_phase_3b.py`. Existing deployed model `best_model.joblib` preserved. Production candidate trained on all 312 observations through FY 2025–26 saved to `models/trained/production_candidate_extended.joblib`.
- **Phase 3C (Completed & Verified):** Recursive multi-year forecast validation (`src/recursive_evaluation.py`). Evaluated multi-step forecast degradation across Horizons 1, 2, and 3 against DPIIT ground truth actuals (FY 2023–24 to FY 2025–26). Demonstrated that Naïve Persistence achieved lowest overall RMSE (12.56 WPI pts) due to post-2022 macroeconomic plateau, while regularized ML pipelines (Ridge RMSE 20.28 WPI pts) prevented explosive divergence of linear momentum baselines (Drift RMSE 66.09 WPI pts). Verified with 10 automated tests in `tests/test_phase_3c.py`.
- **Phase 3D (Completed & Verified):** Dual-trajectory forecasting and farm budgeting architecture implemented and verified. Both ML Trend (Ridge $\alpha=10.0$) and Static Baseline scenarios are rendered in Tab 2 and Tab 3 with transparent side-by-side scenario comparisons, divergence metrics, and mathematical step breakdowns. Verified with 12 automated tests in `tests/test_dual_trajectory.py`. Deployed model artifact `best_model.joblib` SHA-256 strictly preserved.
- **Phase 7B (Pending Authorization — NOT performed):** Future recursive forecasting updates beyond FY 2025–26.
- **Stage 10 (Pending Authorization — NOT performed):** Production deployment cutover.



