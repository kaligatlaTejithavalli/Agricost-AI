# Extended Explainable AI (SHAP) Assumptions, Mechanics, and Verification

**Project:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System  
**Stage:** Phase 3G — Extended Production Explainability  
**Target Variable:** `wpi_index_points` (WPI index points)  
**Base Year:** Base Year 2011-12 = 100.0  
**Model Explained:** Ridge Regression (alpha=10.0) [Production Candidate Extended]  
**Model Artifact:** `C:/Users/kteji/AgriCost-AI/models/trained/production_candidate_extended.joblib` (SHA-256: `0102c1afe2e044c4e11e26bd37ee7720c97a9cd990792ca651f911e7ba5f3f13`)  
**Feature Matrix:** `C:/Users/kteji/AgriCost-AI/data/processed/agricultural_inputs_features_extended.csv` (SHA-256: `ed7899b367d46158001aad2902e3943ba2d96f19d5586af3e6ac2326a3f1e6e2`)  
**Observations Explained:** 312 observations across 26 agricultural input commodities  
**Temporal Coverage:** Financial Years 2014-15 through 2025-26 (12 consecutive financial years)  

---

## 1. Strict Unit Separation & Anti-Rupee Guardrail

- **All SHAP feature attributions and base values are strictly in `WPI index points`.**
- **Never interpret SHAP values as rupee cost variations, retail market price changes, or farmer expenditure adjustments.**
- SHAP explains the mathematical mechanics of the WPI index forecasting pipeline. Translation from wholesale index points to physical rupee expenditure is conducted exclusively downstream by the ratio bridge formula in `src/budget.py`:

  $$\text{Estimated Future Price (₹)} = \text{Current User Price (₹)} \times \left(\frac{\text{Forecast WPI}}{\text{Current WPI}}\right)$$

---

## 2. Explainer Architecture & Mathematical Grounding

- **Explainer Implementation:** `shap.LinearExplainer`
- **Background Masker:** `shap.maskers.Independent` across all 312 transformed training samples ($N=312$).
- **Expected Value (Base Value):** $E[f(X)] = 140.3621$ WPI index points.
- **Exact Additivity Verification:** Because Ridge Regression is a regularized linear model, Shapley values satisfy exact additivity:

  $$E[f(X)] + \sum_{i=1}^{M} \phi_i(x) = f(x) = \hat{y}$$

- Across all 312 samples, the maximum observed additivity error is **5.68e-14 WPI index points**, verifying exact analytical consistency.

---

## 3. Correlated Features & Attribution Dynamics

The model incorporates 12 transformed predictors (5 numerical time-series indicators + 7 one-hot category indicators):
1. `lag_1`: Prior year's WPI index ($y_{t-1}$)
2. `lag_2`: Two years prior WPI index ($y_{t-2}$)
3. `rolling_mean_2`: Two-year moving average ($(y_{t-1} + y_{t-2}) / 2$)
4. `historical_pct_change`: Prior year's percentage index change ($((y_{t-1} - y_{t-2}) / y_{t-2}) \times 100$)
5. `period_order`: Annual chronological trend order index (2 for FY 2014-15 to 13 for FY 2025-26)
6. `input_category`: One-hot encoded category indicators with reference category `Farm Machinery`

### Critical Collinearity Caveat:
- Predictors `lag_1`, `lag_2`, `rolling_mean_2`, and `historical_pct_change` exhibit strong collinearity.
- Under `shap.LinearExplainer`, attributions reflect model coefficient weighting and marginal distributions rather than independent causal levers.
- A feature contribution indicates how the model adjusted its projection given that feature's value relative to the training mean, **NOT a real-world causal guarantee**.
