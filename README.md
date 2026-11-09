# 🌾 AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)
[![SHAP XAI](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen.svg)](https://shap.readthedocs.io/)
[![Pytest Suite](https://img.shields.io/badge/pytest-143%20passed%20%28100%25%29-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Executive Summary & Academic Scope

* **Project Title:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System
* **Academic Level:** Final-Year B.Tech Capstone Project (Computer Science & Engineering / Data Science / Agricultural Informatics)
* **Problem Statement:** Agricultural input costs—fertilizers, crop protection chemicals, diesel fuel, electricity, and farm machinery—fluctuate unpredictably due to macroeconomic shocks, international commodity cycles, supply chain disruptions, and government subsidy interventions. Because smallholder farmers and agribusinesses commit capital months prior to harvest, price volatility severely compresses net farm profitability and impedes pre-sowing capital allocation.
* **Proposed Solution:** AgriCost-AI is an end-to-end, leak-free, explainable machine learning web application that:
  1. Ingests genuine official Wholesale Price Index (WPI) time series from the Government of India (DPIIT, Base Year 2011–12 = 100.0).
  2. Applies leak-free chronological feature engineering (strictly past-lagged indicators).
  3. Evaluates regularized regression models against Naïve Persistence and Drift baselines.
  4. Provides authentic model interpretability using **SHAP (SHapley Additive exPlanations)**.
  5. Implements an **Index-to-Rupee Ratio Bridge** allowing farmers to forecast their specific input budget in Rupees ($₹$) per physical unit and per acre.
  6. Includes a rigorous out-of-sample backtesting validation against newly released official government actuals (FY 2023–24 to FY 2025–26).

---

## 2. Key Architecture & Features

```
                                      AgriCost-AI System Flow
                                      
  [ Official DPIIT WPI Data ] ---> [ Chronological Preprocessor ] ---> [ Leak-Free Feature Engineering ]
     Base 2011-12 = 100.0              Strict Historical Cutoff             Lags (t-1, t-2), Rolling Means
                                                  |                                        |
                                                  v                                        v
  [ Streamlit UI Dashboard ] <--- [ Recursive Horizon Engine ] <--- [ Ridge Regression (alpha=10.0) ]
     4 Core Interactive Tabs           H=1 (2023-24), H=2, H=3              Benchmarked vs Persistence
           |                                      |                                        |
           +---> Tab 1: Price Insights            +---> Tab 3: Budget Planner              v
           +---> Tab 2: Cost Forecast             +---> Tab 4: Explainable AI    [ Authentic SHAP Analysis ]
                                                        (LinearExplainer)          Feature Attribution
```

### The 4 Application Dashboard Tabs

1. **📈 Tab 1: Price Insights (Exploratory Data Analysis):**
   - Interactive historical Wholesale Price Index trajectories from FY 2012–13 to FY 2022–23 for all 26 verified commodities.
   - Key statistical metrics: latest observed WPI, 1-year percentage change, cumulative historical inflation, and volatility.
   - Cross-commodity comparisons and category aggregate time-series plots.
2. **🔮 Tab 2: Cost Forecast (Predictive Engine):**
   - Multi-step recursive forecasting for Horizons 1, 2, and 3 (FY 2023–24, FY 2024–25, and FY 2025–26).
   - Dynamic uncertainty prediction intervals derived from empirical test residuals ($e_h = \pm 1.96 \cdot \text{RMSE} \cdot \sqrt{h}$).
   - Clear model performance card displaying test MAE, test RMSE, and baseline comparisons.
3. **💰 Tab 3: Budget Planner (Decision Support):**
   - Practical farm budget calculator translating index projections into rupee expenditures ($₹$).
   - Prompts the farmer for their current market purchase price ($₹$ per unit), application quantity per acre, and cultivated land size.
   - Computes forecasted unit price, per-acre input cost, total farm budget, and projected rupee variance.
4. **🧠 Tab 4: Explainable AI (XAI with SHAP):**
   - Global feature importance rankings evaluating which historical drivers dictate wholesale price dynamics.
   - Beeswarm distribution summaries and global feature impact bar charts.
   - Local prediction case study (e.g. Urea FY 2022–23) with waterfall plots verifying exact SHAP additivity:
     $$\text{Prediction} = \mathbb{E}[f(X)] + \sum_{i=1}^M \phi_i$$

---

## 3. Data Provenance & Commodity Isolation

### 3.1 Provenance
* **Source:** Office of the Economic Adviser (OEA), Department for Promotion of Industry and Internal Trade (DPIIT), Ministry of Commerce and Industry, Government of India.
* **Official Benchmark:** Wholesale Price Index (WPI), Base Year 2011–12 = 100.0.
* **Temporal Series:** Financial Year averages (April 1 to March 31) spanning FY 2012–13 through FY 2022–23 (11 historical observations per commodity).

### 3.2 26 Verified Agricultural Input Commodities (Across 5 Categories)

| Category | Input Commodities Included | Basket Weight (%) |
| :--- | :--- | :---: |
| **Fertilizers (7)** | Di-ammonium phosphate (DAP), Urea, Mixed fertilizer, Superphosphate, Zinc sulphate, Fertilizers group index | 1.48% |
| **Fuel & Power (3)** | High Speed Diesel (HSD), Petrol, Electricity | 6.81% |
| **Farm Machinery (6)** | Agricultural tractors, Harvesters, Threshers, Power tillers, Agricultural implements, Tractor engines | 0.94% |
| **Crop Protection (5)**| Pesticides, Insecticides, Weedicides, Fungicides, Agrochemical formulations | 0.42% |
| **Livestock Inputs (4)**| Cattle feed compounds, Fodder, Mustard meal / Oilcake feeds, Rice bran extract | 0.58% |

> [!IMPORTANT]
> **Strict Commodity Isolation (Inputs vs. Harvest Outputs):**
> Primary food crops (Wheat, Paddy, Tomato, Onion, Apple, Turmeric) and commercial oilseeds represent farm harvest *revenues*, **not** farm input costs. AgriCost-AI strictly isolates operational farming expenditure commodities. Furthermore, unrecorded informal labour wages are not fabricated.

---

## 4. Mathematical Methodology & Scientific Rigor

### 4.1 Target Variable Definition
The regression target variable is strictly:
$$\text{Target} = \text{WPI index points (dimensionless, normalized to Base 2011–12 = 100.0)}$$
To prevent deceptive representations, index values are **never** labeled as rupees ($₹$).

### 4.2 Leak-Free Time-Series Feature Engineering
All engineered features utilize **strictly past information** ($t-1$ and older):
* **Lag 1:** $y_{t-1}$ (prior year index level)
* **Lag 2:** $y_{t-2}$ (two years prior index level)
* **Historical % Change:** $\Delta_{t-1} = \frac{y_{t-1} - y_{t-2}}{y_{t-2}} \times 100$
* **Rolling Mean 2:** $\bar{y}_{t-1, t-2} = \frac{y_{t-1} + y_{t-2}}{2}$
* **Time Trend:** $\tau_t = t$ (linear integer counter)
* **Categorical Feature:** One-hot encoded `input_category`

### 4.3 Strictly Chronological Train/Test Partitioning
To prevent time-series data leakage:
* **Training Partition:** FY 2014–15 to FY 2019–20 ($N_{\text{train}} = 156$ observations across 26 commodities).
* **Testing Partition (Held-out):** FY 2020–21 to FY 2022–23 ($N_{\text{test}} = 78$ observations across 26 commodities).
* **Zero Shuffle:** Splitting preserves the arrow of time; standardizers and encoders are fitted *exclusively* on the training partition.

### 4.4 Model Selection & Baseline Benchmarking
Ridge Regression ($\alpha = 10.0$) was selected for production because L2 shrinkage controls severe multicollinearity among lagged indicators ($lag_1, lag_2, \text{rolling\_mean}$), preventing runaway oscillations across recursive horizons:

$$\min_{w} \left\{ \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_2^2 \right\}$$

**Benchmark Baselines:**
* **Naïve Persistence:** $\hat{y}_t = y_{t-1}$ (assumes zero change from previous year).
* **Drift Baseline:** $\hat{y}_t = y_{t-1} + \frac{y_{t-1} - y_0}{t-1}$ (extrapolates linear historical momentum).

**Held-Out Test Set Performance (FY 2020–21 to 2022–23):**

| Method | Type | Test MAE (pts) | Test RMSE (pts) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Drift Baseline** | Statistical Baseline | 11.92 | 20.49 | Linear Momentum Benchmark |
| **Ridge Regression ($\alpha=10.0$)** | Machine Learning Pipeline | **12.76** | **21.03** | **Selected Production ML Model** |
| **Naïve Persistence** | Statistical Baseline | 13.88 | 22.02 | Simple Persistence Benchmark |
| **Random Forest Regressor** | Machine Learning | 18.25 | 27.60 | Overfits on small sample time-series |
| **Gradient Boosting Regressor** | Machine Learning | 20.14 | 30.15 | Step-wise tree artifacts in recursion |

### 4.5 The Index-to-Rupee Economic Bridge
To translate macroeconomic wholesale index projections into actionable farmer financial planning:
$$\text{Price}_{\text{future}} = \text{Price}_{\text{current}} \times \left( \frac{\text{WPI}_{\text{forecast}}}{\text{WPI}_{\text{current}}} \right)$$
$$\text{Projected Change (\%)} = \left( \frac{\text{WPI}_{\text{forecast}} - \text{WPI}_{\text{current}}}{\text{WPI}_{\text{current}}} \right) \times 100$$
$$\text{Per-Acre Expenditure} = \text{Price}_{\text{future}} \times \text{Application Rate per Acre}$$
$$\text{Total Farm Budget} = \text{Per-Acre Expenditure} \times \text{Farm Land Area (Acres)}$$

---

## 5. Real-World Backtesting Validation (FY 2023–24 to FY 2025–26)

To assess model generalization in the post-training era, AgriCost-AI conducted an out-of-sample backtesting audit comparing saved Stage 7 recursive multi-year forecasts directly against official DPIIT actuals released in `data/staged/fin_year_index.xls`.

* **Sample Size:** 78 matched prediction-actual pairs across all 26 input commodities and 3 forecast horizons.
* **Overall Metrics:**
  * **Ridge Regression:** MAE = 12.39 pts | RMSE = 20.57 pts | MAPE = 8.52% | **Directional Accuracy = 56.4%**
  * **Naïve Persistence Baseline:** MAE = 9.77 pts | RMSE = 18.27 pts | Directional Accuracy = 0.0%
  * **Drift Baseline:** MAE = 37.13 pts | RMSE = 73.04 pts | Directional Accuracy = 56.4%

### Key Sector Findings:
1. **Fuel & Power Outperformance:**
   Ridge Regression dramatically outperformed both baselines across diesel, petrol, and electricity:
   * **Ridge MAE:** **5.98 pts** | **MAPE:** **3.88%** | **Directional Accuracy:** **77.8%**
   * *Persistence MAE:* 12.74 pts (Ridge is $> 2\times$ more accurate).
   * *Drift MAE:* 98.99 pts (Drift exploded due to post-COVID fuel spikes; Ridge regularized this safely).
2. **Subsidized Price Cap Caveat (Urea):**
   Heavily subsidized commodities subject to statutory Maximum Retail Price (MRP) ceilings (such as Urea) do not obey free-market autoregressive forces. While Ridge projected moderate inflation to 131.6 pts, statutory price freezes held actual WPI at 115.6–116.4 pts. This empirical observation confirms the importance of our project's policy disclaimers.

---

## 6. Directory Structure

```
AgriCost-AI/
├── .streamlit/
│   └── config.toml                 # Agricultural color tokens & headless server settings
├── data/
│   ├── raw/
│   │   ├── INDIAN_GOVT_WHOLESALE_PRICE_INDEX.csv # Official DPIIT historical records
│   │   └── data_dictionary.md      # Provenance audit, field descriptions, and units
│   ├── processed/
│   │   ├── agricultural_inputs_wpi_timeseries.csv # Normalized 26-commodity dataset
│   │   ├── feature_engineered_dataset.csv         # Leak-free lagged feature matrix
│   │   ├── train_features.csv                     # Training split (FY 2014-15 to 2019-20)
│   │   └── test_features.csv                      # Held-out testing split (FY 2020-21 to 2022-23)
│   └── staged/
│       └── fin_year_index.xls      # Benchmark actuals for out-of-sample backtesting
├── models/
│   ├── trained/
│   │   ├── best_model.joblib       # Serialized Ridge Pipeline (StandardScaler + OneHot + Ridge)
│   │   ├── ridge_model.joblib      # Reference baseline pipeline
│   │   └── model_metadata.json     # Architecture records, feature lists, and error bounds
│   └── evaluation/
│       ├── model_comparison.csv    # Error metric ranking (Ridge vs Baselines)
│       ├── category_evaluation.csv # Granular performance by input category
│       ├── test_predictions.csv    # Individual held-out point predictions
│       ├── backtesting/            # Out-of-sample audit reports, tables, and metrics
│       ├── forecasts/              # 3-year recursive projection matrices
│       └── shap/                   # Serialized SHAP values, beeswarms, and waterfall plots
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central path registry and commodity catalog
│   ├── data_loader.py              # Ingestion, schema validation, and provenance assertions
│   ├── preprocessor.py             # Chronological sorting, missing-value and outlier audits
│   ├── feature_engineering.py      # Leak-free lag and rolling feature calculations
│   ├── train.py                    # Chronological split, baseline trainer, Ridge trainer
│   ├── forecast.py                 # Multi-step recursive forecasting engine
│   ├── explainability.py           # Authentic SHAP LinearExplainer and waterfall generator
│   ├── budget.py                   # Index-to-Rupee Ratio Bridge & farm budget calculator
│   └── backtesting.py              # Out-of-sample performance evaluation engine
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py         # Schema, provenance, and commodity tests
│   ├── test_preprocessor.py        # Outlier and sort integrity tests
│   ├── test_feature_engineering.py # Leakage prevention and lag boundary tests
│   ├── test_train.py               # Model fitting, baseline mathematical correctness tests
│   ├── test_explainability.py      # SHAP additivity and feature attribution tests
│   ├── test_forecast.py            # Recursive projection and uncertainty interval tests
│   ├── test_budget.py              # Ratio bridge and budget arithmetic tests
│   ├── test_app.py                 # Streamlit AppTest end-to-end UI tests
│   └── test_backtesting.py         # Real-world backtesting verification tests
├── app.py                          # Multi-tab Streamlit interactive web application
├── requirements.txt                # Pinned production dependencies
├── runtime.txt                     # Target Python runtime (python-3.11)
├── PROJECT_PLAN.md                 # Foundational architecture specification
└── README.md                       # Comprehensive capstone documentation & viva guide
```

---

## 7. Installation, Setup & Execution

### 7.1 Prerequisites
* Python 3.11, 3.12, 3.13, or 3.14
* Git

### 7.2 Local Installation
```bash
# 1. Clone repository
git clone https://github.com/kaligatlaTejithavalli/Fraud-detection.git AgriCost-AI
cd AgriCost-AI

# 2. Create and activate a clean virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### 7.3 Running the Automated Test Suite
AgriCost-AI incorporates comprehensive automated unit and integration tests:
```bash
pytest -v
```
**Expected Outcome:** `143 passed in ~15s (100% success rate, 0 failures)`.

### 7.4 Launching the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to access the interactive dashboard.

---
### 7.5 Accessing the Live Application

AgriCost-AI is deployed and publicly accessible through Streamlit Community Cloud.

**Live Application:** [Open AgriCost-AI](https://agricost-ai-5l6myfzgtlgbpprew9uegz.streamlit.app/)

Users can explore historical agricultural input price trends, generate future WPI forecasts, estimate farming budgets, and understand model predictions through SHAP-based explanations.

**Note:** The application may take a few moments to load if it has been inactive.

## 8. Comprehensive Academic Viva & Defense Guide

Examiners frequently ask targeted questions about machine learning rigor, time-series mechanics, data leakage, and real-world agricultural economics. Below are the definitive answers implemented in AgriCost-AI:

### Q1: Why did you forecast Wholesale Price Index (WPI) points instead of direct Rupee market prices?
> **Answer:** Indian retail farmgate prices vary widely across local mandis, transport distances, and dealer margins; there is no centralized, legally verified national retail time series covering multi-decade input costs. In contrast, the Government of India (DPIIT) publishes the official Wholesale Price Index (Base 2011–12 = 100.0), which provides standard, audited macroeconomic tracking. To bridge this index to actual farming costs, we formulated the **Index-to-Rupee Ratio Bridge**, allowing farmers to input their local baseline price in rupees ($₹$) and project future expenses via verified index growth ratios.

### Q2: How did you guarantee zero data leakage in your time-series pipeline?
> **Answer:** We enforced three strict architectural constraints:
> 1. **Chronological Splitting:** We strictly rejected `train_test_split(shuffle=True)`. Training was fitted exclusively on earlier years (FY 2014–15 to 2019–20), while evaluation was held out on future years (FY 2020–21 to 2022–23).
> 2. **Preprocessing Isolation:** Scalers (`StandardScaler`) and encoders (`OneHotEncoder`) were fitted *only* on the training split and transformed on test data without refitting.
> 3. **Lagged Feature Generation:** All rolling averages and change metrics were computed using $t-1$ and $t-2$ values; no current-timestep or forward-looking information was included in feature rows.

### Q3: Why choose Ridge Regression over Deep Learning (LSTM / Transformers)?
> **Answer:** Official national macroeconomic time series provide annual observations (11 financial years in the 2011–12 base series). Deep neural networks (LSTM, GRU, Temporal Transformers) possess tens of thousands of parameters and require large-sample high-frequency data; applying them to small sample time-series results in catastrophic overfitting. Ridge Regression ($\alpha = 10.0$) introduces an analytical L2 penalty that regularizes collinear lag features, guarantees stable deterministic solutions, and enables authentic SHAP interpretability.

### Q4: Why did Urea's Wholesale Price Index remain virtually flat over 10 years?
> **Answer:** Urea is classified as a statutorily controlled commodity under the Fertilizer Control Order (FCO) in India. The Central Government fixes the Maximum Retail Price (MRP) of Urea at a statutory subsidized rate (~₹242 per 45kg bag), absorbing wholesale cost surges through direct manufacturer subsidies. In contrast, non-urea fertilizers like DAP and MOP are decontrolled under the Nutrient Based Subsidy (NBS) scheme and fluctuate with international raw material markets. AgriCost-AI explicitly documents this policy mechanism.

### Q5: How does SHAP interpret correlated features ($lag_1, lag_2, \text{rolling\_mean}$)?
> **Answer:** Because lag features are mathematically collinear, `shap.LinearExplainer` evaluates feature attributions based on the model's regularized weights and empirical training distributions. SHAP measures the marginal contribution of each feature to the model's specific projection, **not causal independence in the real world**. We incorporate an explicit technical disclaimer explaining that a high SHAP value for $lag_1$ signifies heavy model reliance on prior-year index momentum.

### Q6: What is the difference between single-step test evaluation and multi-step recursive forecasting?
> **Answer:** In single-step test evaluation (FY 2020–21 to 2022–23), the model predicts $y_t$ using the *actual* historical lags from $t-1$. In multi-step recursive forward forecasting (Horizons 1, 2, and 3), future true lags do not yet exist, so the model feeds its own prior prediction back into subsequent input lag vectors ($\hat{y}_{t+1} \to \text{lag}_1$). Error naturally compounds across time ($\sigma_h \propto \sqrt{h}$), which is why AgriCost-AI visualizes expanding confidence bands for distant horizons.

---

## 9. Academic Conclusions & Limitations

1. **Wholesale vs. Retail Farmgate Spread:** WPI captures bulk wholesale trade; localized retail dealer margins or transport bottlenecks may cause short-term farmgate divergence.
2. **Policy Exogenous Shocks:** Sudden regulatory changes (subsidy hikes, export bans, statutory price freezes) cannot be inferred from historical price autoregression alone.
3. **Recursive Horizon Compounding:** Forward recursive predictions compound variance over 3+ year horizons, emphasizing the necessity of presenting forecasts as statistical planning bands rather than absolute certainties.

---

*AgriCost-AI — Developed as an Academic Capstone Project in Applied Machine Learning & Agricultural Decision Support Systems.*
