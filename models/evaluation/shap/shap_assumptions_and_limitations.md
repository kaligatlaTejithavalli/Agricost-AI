# Explainable AI (SHAP) Assumptions, Mechanics, and Limitations

**Project:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System  
**Stage:** Stage 6 — Explainable AI (XAI) using SHAP  
**Target Variable:** `wpi_index_points` (WPI index points)  
**Base Year:** Base Year 2011-12 = 100.0  
**Model Explained:** Ridge Regression (alpha = 10.0)  

---

## 1. Unit Semantics & Anti-Rupee Guardrail

- **All SHAP feature contributions and base values are strictly in `WPI index points`.**
- **Never interpret SHAP values as rupee price movements, actual retail fertilizer costs, farm machinery hiring rates, or guaranteed cash budget impacts.**
- SHAP explains the internal mechanics of the WPI index forecasting pipeline. The translation from WPI index points to user rupee expenditure is handled downstream by the budget planning bridge formula:

  Estimated Future Price (Rs) = Current User Price (Rs) * (Forecasted WPI / Current WPI)

---

## 2. Explainer Architecture & Mathematical Grounding

- **Explainer Type:** `shap.LinearExplainer`
- **Background Masker:** `shap.maskers.Independent` fitted strictly on the **training split** (156 samples, 2014-15 to 2019-20).
- **Exact Additivity:** Because Ridge Regression is a regularized linear model, `LinearExplainer` produces exact analytical Shapley attributions:

  E[f(X)] + sum(phi_i) = f(x) = y_hat

  This additivity holds within floating-point machine precision (< 1e-10 WPI index points).

---

## 3. Correlated Features & Attribution Dynamics

The AgriCost-AI feature pipeline incorporates closely related historical time series predictors:
1. `lag_1`: Prior year's WPI index (y_{t-1})
2. `lag_2`: Two years prior WPI index (y_{t-2})
3. `rolling_mean_2`: Two-year moving average ((y_{t-1} + y_{t-2}) / 2)
4. `historical_pct_change`: Prior year's percentage index change (((y_{t-1} - y_{t-2}) / y_{t-2}) * 100)

### Critical Interpretive Caveat:
- These four variables exhibit strong mutual collinearity.
- `shap.LinearExplainer` with an independent masker calculates attributions based on linear model coefficients and the marginal background distribution.
- **Attributions among collinear variables are shared according to the model's regularized weights, not independent real-world causality.**
- A negative SHAP value for `lag_1` in a specific prediction does **NOT** imply that higher historical prices physically lower future prices; rather, it reflects how the model balances `lag_1`, `rolling_mean_2`, and regularized mean reversion relative to the training baseline.
- **Do not interpret SHAP attributions as causal economic levers.**

---

## 4. Categorical Feature Attribution

- Commodity category is encoded via `OneHotEncoder(drop='first')` across the 6 major input categories.
- SHAP evaluates the contribution of category indicators relative to the reference category (`Farm Machinery`).
- Category contributions capture category-wide structural price level shifts relative to the broad agricultural input basket.

---

## 5. Scope of Held-out Evaluation

- Explanations are computed across all **78 held-out test observations** spanning financial years 2020-21, 2021-22, and 2022-23 across all 26 commodities.
- The test split was strictly evaluated using the fitted preprocessing pipeline without any refitting or data leakage.
