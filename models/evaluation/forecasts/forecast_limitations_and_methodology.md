# Future WPI Forecasting Engine: Methodology, Recursive Updates, and Limitations

**Project:** AgriCost-AI: AI-Based Agricultural Input Cost Forecasting & Decision Support System  
**Stage:** Stage 7 — Future WPI Forecasting Engine  
**Target Variable:** `wpi_index_points` (WPI index points, Base Year 2011-12 = 100.0)  
**Forecasting Model:** Ridge Regression (alpha = 10.0, fitted pipeline)  
**Latest Observed Year:** Financial Year 2022-23 (DPIIT official release)  
**Supported Horizons:** 1 to 3 Financial Years (2023-24, 2024-25, 2025-26)  

---

## 1. Recursive Forecasting Mechanics

The model forecasts wholesale price indices one annual step at a time. For horizons beyond one step,
predictions are fed back recursively as lag inputs for subsequent years:

- **Horizon 1 (FY 2023-24):**
  - `lag_1` = Actual WPI (2022-23)
  - `lag_2` = Actual WPI (2021-22)
  - `rolling_mean_2` = (lag_1 + lag_2) / 2
  - `historical_pct_change` = ((lag_1 - lag_2) / lag_2) * 100
  - `period_order` = 11
  - Predicts y_hat(2023-24)

- **Horizon 2 (FY 2024-25):**
  - `lag_1` = Predicted y_hat(2023-24)
  - `lag_2` = Actual WPI (2022-23)
  - `rolling_mean_2` = (y_hat(2023-24) + WPI(2022-23)) / 2
  - `historical_pct_change` = ((y_hat(2023-24) - WPI(2022-23)) / WPI(2022-23)) * 100
  - `period_order` = 12
  - Predicts y_hat(2024-25)

- **Horizon 3 (FY 2025-26):**
  - `lag_1` = Predicted y_hat(2024-25)
  - `lag_2` = Predicted y_hat(2023-24)
  - `rolling_mean_2` = (y_hat(2024-25) + y_hat(2023-24)) / 2
  - `historical_pct_change` = ((y_hat(2024-25) - y_hat(2023-24)) / y_hat(2023-24)) * 100
  - `period_order` = 13
  - Predicts y_hat(2025-26)

---

## 2. Uncertainty Compounding & Prediction Intervals

- **One-Step Test RMSE:** The held-out test evaluation RMSE was 21.03 WPI index points.
- **Compounding Error:** In multi-step recursive forecasting, prediction errors in step h=1 propogate
  into lag features for h=2 and h=3. As a result, forecast variance expands over time.
- **Indicative Planning Interval:** We provide an indicative 80% planning interval based on analytical
  standard error expansion: sigma_h = RMSE * sqrt(h). These intervals indicate increasing dispersion,
  not a deterministic guarantee of economic prices.

---

## 3. Historical Data Cutoff & Temporal Baseline Caveat

- The government dataset terminates in FY 2022-23.
- Forecasts for FY 2023-24, 2024-25, and 2025-26 are statistical projections, not official published indices.
- When applying the budget bridge formula:
  $$\text{Estimated Future Price (Rs)} = \text{Current User Price (Rs)} \times \left( \frac{\text{Forecast WPI}}{\text{Current WPI}} \right)$$
  users must note that the baseline Current WPI corresponds to FY 2022-23. If the user provides a price from a later
  calendar year, the relative ratio projects inflation from the 2022-23 baseline level.

---

## 4. Anti-Rupee Guardrail

- WPI forecasts are dimensionless statistical index points (2011-12 = 100.0).
- They must NEVER be displayed as rupees per unit or farmer expenditure totals.
