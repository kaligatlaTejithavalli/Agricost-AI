# Phase 3C: Recursive Multi-Year Forecast Validation Report
**Evaluation Scope:** Retrospective validation of recursive multi-year forecasts across 26 agricultural input commodities for Horizons 1, 2, and 3 (FY 2023–24, FY 2024–25, and FY 2025–26).  
**Historical Cutoff:** 2022-23 (period_order = 10)  
**Target Unit:** WPI index points (Base Year 2011-12 = 100.0)  
**Total Evaluated Observations:** 78 pairs (26 commodities × 3 horizons)  

---

## 1. Overall Recursive Model Comparison

| Model / Baseline | Model Type | Overall MAE (WPI pts) | Overall RMSE (WPI pts) | Overall MAPE (%) | Directional Acc. (%) | Beats Naïve? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naïve Persistence Baseline** | Baseline | 8.96 | 12.56 | 5.54% | 0.0% | No |
| **Rolling Mean Baseline (2-Year MA)** | Baseline | 11.18 | 17.38 | 6.31% | 48.7% | No |
| **Stage 7 Saved Ridge Regression (alpha=10.0)** | Machine Learning | 12.53 | 20.28 | 7.30% | 57.7% | No |
| **Retrospective Ridge Regression (alpha=1.0)** | Machine Learning | 25.36 | 29.80 | 16.96% | 47.4% | No |
| **Retrospective Ridge Regression (alpha=10.0)** | Machine Learning | 26.46 | 30.45 | 17.76% | 47.4% | No |
| **Random Forest Regressor (depth=3)** | Machine Learning | 22.10 | 33.00 | 13.07% | 48.7% | No |
| **Gradient Boosting Regressor (depth=2)** | Machine Learning | 23.33 | 33.46 | 14.19% | 47.4% | No |
| **Drift Baseline (Linear Momentum)** | Baseline | 44.14 | 66.09 | 25.73% | 51.3% | No |

---

## 2. Performance by Forecast Horizon

| Horizon | Financial Year | Model | MAE (WPI pts) | RMSE (WPI pts) | MAPE (%) | Directional Acc. (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| H=1 | 2023-24 | Naïve Persistence Baseline | 8.09 | 11.65 | 4.83% | 0.0% |
| H=2 | 2024-25 | Naïve Persistence Baseline | 8.73 | 12.36 | 5.43% | 0.0% |
| H=3 | 2025-26 | Naïve Persistence Baseline | 10.07 | 13.61 | 6.35% | 0.0% |
| H=1 | 2023-24 | Rolling Mean Baseline (2-Year MA) | 12.26 | 19.96 | 6.59% | 50.0% |
| H=2 | 2024-25 | Rolling Mean Baseline (2-Year MA) | 9.42 | 14.54 | 5.51% | 50.0% |
| H=3 | 2025-26 | Rolling Mean Baseline (2-Year MA) | 11.85 | 17.22 | 6.83% | 46.1% |
| H=1 | 2023-24 | Stage 7 Saved Ridge Regression (alpha=10.0) | 10.12 | 16.59 | 5.79% | 50.0% |
| H=2 | 2024-25 | Stage 7 Saved Ridge Regression (alpha=10.0) | 12.09 | 19.84 | 7.15% | 57.7% |
| H=3 | 2025-26 | Stage 7 Saved Ridge Regression (alpha=10.0) | 15.38 | 23.77 | 8.97% | 65.4% |
| H=1 | 2023-24 | Retrospective Ridge Regression (alpha=1.0) | 15.64 | 19.56 | 10.18% | 46.1% |
| H=2 | 2024-25 | Retrospective Ridge Regression (alpha=1.0) | 25.98 | 29.17 | 17.35% | 46.1% |
| H=3 | 2025-26 | Retrospective Ridge Regression (alpha=1.0) | 34.44 | 37.81 | 23.36% | 50.0% |
| H=1 | 2023-24 | Retrospective Ridge Regression (alpha=10.0) | 15.84 | 19.35 | 10.35% | 46.1% |
| H=2 | 2024-25 | Retrospective Ridge Regression (alpha=10.0) | 26.98 | 29.47 | 18.07% | 46.1% |
| H=3 | 2025-26 | Retrospective Ridge Regression (alpha=10.0) | 36.57 | 39.24 | 24.86% | 50.0% |
| H=1 | 2023-24 | Random Forest Regressor (depth=3) | 14.48 | 26.86 | 7.93% | 42.3% |
| H=2 | 2024-25 | Random Forest Regressor (depth=3) | 22.36 | 32.44 | 13.17% | 50.0% |
| H=3 | 2025-26 | Random Forest Regressor (depth=3) | 29.46 | 38.64 | 18.10% | 53.9% |
| H=1 | 2023-24 | Gradient Boosting Regressor (depth=2) | 15.96 | 25.38 | 9.16% | 38.5% |
| H=2 | 2024-25 | Gradient Boosting Regressor (depth=2) | 24.06 | 34.75 | 14.45% | 50.0% |
| H=3 | 2025-26 | Gradient Boosting Regressor (depth=2) | 29.96 | 38.83 | 18.96% | 53.9% |
| H=1 | 2023-24 | Drift Baseline (Linear Momentum) | 23.12 | 31.00 | 13.65% | 50.0% |
| H=2 | 2024-25 | Drift Baseline (Linear Momentum) | 44.13 | 61.12 | 25.70% | 50.0% |
| H=3 | 2025-26 | Drift Baseline (Linear Momentum) | 65.17 | 91.70 | 37.84% | 53.9% |

---

## 3. Key Findings & Empirical Conclusions

1. **Does the ML Model Outperform Naïve Persistence?**  

   **No.** In recursive multi-year point forecasting, **Naïve Persistence achieved the lowest overall error** (RMSE = 12.56 WPI pts vs. 20.28 WPI pts for Stage 7 Ridge).  
   This occurred because the official Government of India WPI index experienced an unexpected macro plateau following the high-inflation period of FY 2021–22 to FY 2022–23. In a flat market, carrying forward the last known value avoids error accumulation.

2. **Compounding Uncertainty in Multi-Step Recursive Projections:**  

   For all machine learning models, error expanded consistently across horizons ($H1 \to H2 \to H3$). For example, Stage 7 Ridge RMSE expanded from 16.59 pts (H1) to 19.84 pts (H2) and 23.77 pts (H3), confirming that replacing true lags with predicted values compounds uncertainty.

3. **Protection Against Explosive Divergence (Drift Baseline):**  

   Machine learning models decisively outperformed the Drift baseline (20.28 pts vs. 66.09 pts). Pure linear momentum exploded catastrophically, while L2 regularization constrained the slope.

4. **Methodological Honesty Regarding Production Candidate:**  

   The production candidate (`production_candidate_extended.joblib`) was trained on all 312 observations through FY 2025–26. Predicting FY 2023–24 through FY 2025–26 using that model would represent in-sample evaluation with target leakage. Our retrospective evaluation strictly used models fitted on data available at the FY 2022–23 cutoff.

---

*Generated by AgriCost-AI Phase 3C Recursive Multi-Year Forecast Validation Engine.*