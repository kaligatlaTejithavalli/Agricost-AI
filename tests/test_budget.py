"""Unit tests for the budget estimation module (src/budget.py).

Verifies:
1. Index-ratio calculation accuracy.
2. Percentage-change calculation accuracy.
3. Rejection of invalid inputs:
   - Missing (None, NaN) WPI
   - Zero or negative WPI
   - Non-numeric WPI
   - Invalid / negative / zero current user price
   - Empty or missing price unit
4. Rejection of mismatched commodity series (using unrelated commodities).
5. Rejection of mismatched series types (mixing FY and CY).
6. Transparent labels and methodology explanations.
7. Budget Planner input expenditure calculations with explicit usage quantities and units.
8. Rejection of unparameterized usage assumptions.
"""

import math
import pytest

from src.budget import (
    calculate_farm_budget,
    calculate_input_expenditure,
    estimate_future_input_price,
    validate_numeric_positive,
    FarmBudgetPlan,
    FarmExpenditureItem,
    PriceProjectionResult,
)
from src.config import (
    LABEL_ESTIMATED_FUTURE_PRICE,
    LABEL_ESTIMATED_PCT_CHANGE,
    LABEL_FORECAST_WPI,
    LABEL_HISTORICAL_WPI,
    LABEL_USER_CURRENT_PRICE,
    TARGET_LABEL,
)


class TestIndexRatioAndPercentageCalculation:
    """Tests for core mathematical index-ratio and percentage change calculations."""

    def test_standard_index_inflation(self):
        """Test calculation when WPI increases (inflationary scenario)."""
        # Example: DAP increases from 145.3 to 164.0 index points (+12.87%)
        # User current price: ₹1350 per 50 kg bag
        result = estimate_future_input_price(
            current_wpi=145.3,
            forecast_wpi=164.0,
            current_user_price=1350.0,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020002,
            commodity_name="Di ammonium phosphate",
            current_period="2021-22",
            forecast_period="2022-23",
            series_type="FY",
        )

        expected_ratio = 164.0 / 145.3
        expected_future_price = round(1350.0 * expected_ratio, 2)
        expected_pct_change = round(((164.0 - 145.3) / 145.3) * 100.0, 2)

        assert result.estimated_future_price == expected_future_price
        assert result.projected_percentage_change == expected_pct_change
        assert result.estimated_future_price == 1523.74
        assert result.projected_percentage_change == 12.87

    def test_index_deflation(self):
        """Test calculation when WPI decreases (deflationary scenario)."""
        # Example: Diesel index drops from 120.0 to 108.0 (-10.0%)
        # User current price: ₹95 per litre
        result = estimate_future_input_price(
            current_wpi=120.0,
            forecast_wpi=108.0,
            current_user_price=95.0,
            price_unit="₹ per litre",
            commodity_code=1202000005,
            commodity_name="HSD",
        )

        assert result.estimated_future_price == 85.50
        assert result.projected_percentage_change == -10.0

    def test_zero_index_change(self):
        """Test calculation when forecast WPI equals current WPI."""
        result = estimate_future_input_price(
            current_wpi=150.0,
            forecast_wpi=150.0,
            current_user_price=2500.0,
            price_unit="₹ per quintal",
        )

        assert result.estimated_future_price == 2500.0
        assert result.projected_percentage_change == 0.0


class TestInvalidInputsValidation:
    """Tests ensuring invalid, non-numeric, or non-positive inputs are rejected."""

    def test_reject_zero_current_wpi(self):
        """Reject current_wpi <= 0 (division by zero / undefined index)."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=0,
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_negative_current_wpi(self):
        """Reject negative current_wpi."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=-115.0,
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_zero_forecast_wpi(self):
        """Reject forecast_wpi <= 0."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=0.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_negative_forecast_wpi(self):
        """Reject negative forecast_wpi."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=-120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_nan_or_none_wpi(self):
        """Reject NaN or None WPI values."""
        with pytest.raises(ValueError, match="cannot be None or missing"):
            estimate_future_input_price(
                current_wpi=None,
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

        with pytest.raises(ValueError, match="cannot be NaN"):
            estimate_future_input_price(
                current_wpi=float("nan"),
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_non_numeric_wpi(self):
        """Reject non-numeric WPI string."""
        with pytest.raises(ValueError, match="must be a numeric value"):
            estimate_future_input_price(
                current_wpi="one-hundred",
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="₹ per bag",
            )

    def test_reject_zero_or_negative_user_price(self):
        """Reject user price <= 0."""
        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=120.0,
                current_user_price=0.0,
                price_unit="₹ per bag",
            )

        with pytest.raises(ValueError, match="must be strictly positive"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=120.0,
                current_user_price=-450.0,
                price_unit="₹ per bag",
            )

    def test_reject_missing_or_empty_price_unit(self):
        """Reject missing or whitespace-only physical price unit."""
        with pytest.raises(ValueError, match="Invalid physical unit"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="",
            )

        with pytest.raises(ValueError, match="Invalid physical unit"):
            estimate_future_input_price(
                current_wpi=115.0,
                forecast_wpi=120.0,
                current_user_price=1000.0,
                price_unit="   ",
            )


class TestSeriesCompatibilityAndMismatches:
    """Tests ensuring mismatched commodities or incompatible series are blocked."""

    def test_reject_mismatched_commodity_codes(self):
        """Ensure current and forecast WPI cannot refer to different commodities."""
        # Trying to use Urea current WPI with DAP forecast WPI
        with pytest.raises(ValueError, match="Commodity series mismatch"):
            estimate_future_input_price(
                current_wpi=110.1,
                forecast_wpi=164.0,
                current_user_price=268.0,
                price_unit="₹ per 45 kg bag",
                commodity_code=1310020001,  # Urea
                forecast_commodity_code=1310020002,  # DAP
            )

    def test_reject_mismatched_series_types(self):
        """Ensure Financial Year (FY) series cannot be compared directly with Calendar Year (CY)."""
        with pytest.raises(ValueError, match="Series type mismatch"):
            estimate_future_input_price(
                current_wpi=110.1,
                forecast_wpi=125.0,
                current_user_price=268.0,
                price_unit="₹ per 45 kg bag",
                series_type="FY",
                forecast_series_type="CY",
            )


class TestTransparentLabelsAndExplanations:
    """Tests verifying Task 4 labeling requirements."""

    def test_labels_and_disclaimers_present(self):
        """Verify that output contains distinct labels for all required quantities."""
        result = estimate_future_input_price(
            current_wpi=120.0,
            forecast_wpi=132.0,
            current_user_price=1000.0,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020002,
            commodity_name="Di ammonium phosphate",
        )

        # Explicit target unit check (Task 2)
        assert result.target_unit == TARGET_LABEL
        assert result.target_unit == "WPI index points"

        # Explicit labels dictionary checks (Task 4)
        labels = result.labels
        assert labels["historical_wpi"] == LABEL_HISTORICAL_WPI
        assert labels["forecast_wpi"] == LABEL_FORECAST_WPI
        assert labels["user_current_price"] == LABEL_USER_CURRENT_PRICE
        assert labels["estimated_future_price"] == LABEL_ESTIMATED_FUTURE_PRICE
        assert labels["projected_percentage_change"] == LABEL_ESTIMATED_PCT_CHANGE

        # Transparent explanation & disclaimer checks
        assert "estimated_future_price = current_user_price * (forecast_wpi / current_wpi)" in result.methodology_explanation
        assert "NOT a guaranteed market quotation" in result.assumptions_disclaimer
        assert result.is_estimate is True


class TestBudgetPlannerCalculations:
    """Tests verifying Task 5 Budget Planner rules."""

    def test_expenditure_with_per_acre_quantity(self):
        """Calculate expenditure using quantity per acre and farm size."""
        # 5 acres, 2 bags/acre = 10 bags total
        # Unit price: ₹1350 current, estimated future ₹1500
        projection = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=111.111,
            current_user_price=1350.0,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020002,
            commodity_name="DAP",
        )
        assert projection.estimated_future_price == 1500.0

        item = calculate_input_expenditure(
            price_projection=projection,
            quantity_per_acre=2.0,
            farm_size_acres=5.0,
            quantity_unit="50 kg bags",
        )

        assert item.total_quantity == 10.0
        assert item.current_total_cost == 13500.0
        assert item.estimated_future_total_cost == 15000.0
        assert item.estimated_cost_increase == 1500.0
        assert item.quantity_unit == "50 kg bags"

    def test_expenditure_with_direct_total_quantity(self):
        """Calculate expenditure using direct total quantity across farm."""
        projection = estimate_future_input_price(
            current_wpi=120.0,
            forecast_wpi=132.0,
            current_user_price=90.0,
            price_unit="₹ per litre",
            commodity_code=1202000005,
            commodity_name="HSD",
        )
        assert projection.estimated_future_price == 99.0

        item = calculate_input_expenditure(
            price_projection=projection,
            total_quantity=200.0,
            quantity_unit="litres",
        )

        assert item.total_quantity == 200.0
        assert item.current_total_cost == 18000.0
        assert item.estimated_future_total_cost == 19800.0
        assert item.estimated_cost_increase == 1800.0

    def test_reject_missing_quantities(self):
        """Reject calculation when usage quantity is omitted (do not invent usage rates)."""
        projection = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=110.0,
            current_user_price=1000.0,
            price_unit="₹ per bag",
        )

        with pytest.raises(ValueError, match="Missing usage quantity"):
            calculate_input_expenditure(price_projection=projection)

    def test_reject_zero_or_negative_quantities(self):
        """Reject non-positive quantities or acreage."""
        projection = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=110.0,
            current_user_price=1000.0,
            price_unit="₹ per bag",
        )

        with pytest.raises(ValueError, match="must be strictly positive"):
            calculate_input_expenditure(
                price_projection=projection,
                total_quantity=0,
            )

        with pytest.raises(ValueError, match="must be strictly positive"):
            calculate_input_expenditure(
                price_projection=projection,
                quantity_per_acre=2.0,
                farm_size_acres=-5.0,
            )

    def test_aggregate_farm_budget_plan(self):
        """Test complete budget aggregation across multiple inputs."""
        # Input 1: Fertilizer (DAP)
        proj_dap = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=110.0,
            current_user_price=1350.0,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020002,
            commodity_name="DAP",
        )
        item_dap = calculate_input_expenditure(
            price_projection=proj_dap,
            quantity_per_acre=2.0,
            farm_size_acres=5.0,
            quantity_unit="bags",
        )

        # Input 2: Diesel (HSD)
        proj_diesel = estimate_future_input_price(
            current_wpi=100.0,
            forecast_wpi=120.0,
            current_user_price=90.0,
            price_unit="₹ per litre",
            commodity_code=1202000005,
            commodity_name="HSD",
        )
        item_diesel = calculate_input_expenditure(
            price_projection=proj_diesel,
            quantity_per_acre=30.0,
            farm_size_acres=5.0,
            quantity_unit="litres",
        )

        budget_plan = calculate_farm_budget(
            items=[item_dap, item_diesel],
            farm_size_acres=5.0,
        )

        # Item 1: DAP: 10 bags * 1350 = 13500 current; 10 bags * 1485 = 14850 future
        # Item 2: Diesel: 150 litres * 90 = 13500 current; 150 litres * 108 = 16200 future
        assert budget_plan.current_total_budget == 27000.0
        assert budget_plan.estimated_future_total_budget == 31050.0
        assert budget_plan.projected_net_change == 4050.0
        assert budget_plan.overall_percentage_change == 15.0
        assert len(budget_plan.items) == 2


# ---------------------------------------------------------------------------
# Test Class: Unit Label Formatting and Precision (Phase 3I Regression)
# ---------------------------------------------------------------------------
class TestUnitLabelFormattingAndPrecision:
    """Verifies separation of physical units, price units, and full-precision calculations."""

    def test_extract_physical_unit_across_all_commodities(self):
        """Verify extract_physical_unit produces clean physical units across all 26 catalog items."""
        from src.budget import extract_physical_unit
        from src.config import AGRICULTURAL_INPUT_CATALOG

        assert len(AGRICULTURAL_INPUT_CATALOG) == 26
        for code, meta in AGRICULTURAL_INPUT_CATALOG.items():
            phys = extract_physical_unit(meta.suggested_unit)
            assert "₹" not in phys, f"Currency symbol found in physical unit: {phys} for {meta.name}"
            assert "Rs" not in phys, f"Rupee abbreviation found in physical unit: {phys} for {meta.name}"
            assert "per" not in phys.lower(), f"'per' found in physical unit: {phys} for {meta.name}"
            assert "rate" not in phys.lower(), f"'rate' found in physical unit: {phys} for {meta.name}"
            assert len(phys) > 0, f"Empty physical unit for {meta.name}"

        # Verify key individual commodities
        assert extract_physical_unit("₹ per 50 kg bag") == "50 kg bag"
        assert extract_physical_unit("₹ per litre") == "litre"
        assert extract_physical_unit("₹ per kWh") == "kWh"
        assert extract_physical_unit("₹ per hour (hiring rate)") == "hour"
        assert extract_physical_unit("₹ per acre (harvest hiring rate)") == "acre"
        assert extract_physical_unit("₹ per quintal (threshing rate)") == "quintal"
        assert extract_physical_unit("₹ per tyre") == "tyre"
        assert extract_physical_unit("50 kg bag") == "50 kg bag"
        assert extract_physical_unit("bags") == "bags"

    def test_format_price_unit_prevents_duplication(self):
        """Verify format_price_unit creates valid price unit without duplicating currency or 'per'."""
        from src.budget import format_price_unit

        assert format_price_unit("50 kg bag") == "₹ per 50 kg bag"
        assert format_price_unit("₹ per 50 kg bag") == "₹ per 50 kg bag"
        assert format_price_unit("₹ per ₹ per 50 kg bag") == "₹ per 50 kg bag"
        assert format_price_unit("litre") == "₹ per litre"
        assert format_price_unit("₹ per litre") == "₹ per litre"
        assert format_price_unit("₹/litre") == "₹ per litre"
        assert format_price_unit("kWh") == "₹ per kWh"
        assert format_price_unit("hour") == "₹ per hour"

    def test_format_quantity_unit(self):
        """Verify format_quantity_unit produces natural quantity strings without currency."""
        from src.budget import format_quantity_unit

        # Multi-unit bags
        assert format_quantity_unit(20.0, "50 kg bag") == "20 bags"
        assert format_quantity_unit(1.0, "50 kg bag") == "1 50 kg bag"
        assert format_quantity_unit(20.0, "bags") == "20 bags"
        assert format_quantity_unit(1.0, "bag") == "1 bag"

        # Liquid volume
        assert format_quantity_unit(10.0, "litre") == "10 litres"
        assert format_quantity_unit(1.0, "litre") == "1 litre"

        # Electricity
        assert format_quantity_unit(5.0, "kWh") == "5 kWh"

        # Fractional quantities
        assert format_quantity_unit(2.5, "litre") == "2.50 litres"

    def test_prevention_of_duplicated_rupee_symbols_in_expenditure_items(self):
        """Verify FarmExpenditureItem.quantity_unit strictly contains no currency symbols."""
        from src.budget import (
            calculate_input_expenditure,
            estimate_future_input_price,
            extract_physical_unit,
        )

        proj = estimate_future_input_price(
            current_wpi=115.60,
            forecast_wpi=122.9643,
            current_user_price=266.50,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020001,
            commodity_name="Urea",
        )

        # Default fallback to projection physical unit
        item_default = calculate_input_expenditure(
            price_projection=proj,
            total_quantity=20.0,
        )
        assert item_default.quantity_unit == "50 kg bag"
        assert "₹" not in item_default.quantity_unit

        # Even if caller passes a price unit string as quantity_unit, it must be cleaned
        item_cleaned = calculate_input_expenditure(
            price_projection=proj,
            total_quantity=20.0,
            quantity_unit="₹ per 50 kg bag",
        )
        assert item_cleaned.quantity_unit == "50 kg bag"
        assert "₹" not in item_cleaned.quantity_unit

        # Explicit clean quantity unit preserved
        item_explicit = calculate_input_expenditure(
            price_projection=proj,
            total_quantity=20.0,
            quantity_unit="bags",
        )
        assert item_explicit.quantity_unit == "bags"
        assert "₹" not in item_explicit.quantity_unit

    def test_exact_urea_precision_calculation(self):
        """Verify full mathematical precision for Urea FY 2026-27 with baseline FY 2025-26."""
        from src.budget import (
            calculate_dual_scenario_expenditure,
            estimate_dual_scenario_input_price,
        )

        # Exact unrounded parameters
        base_wpi = 115.60
        unrounded_ridge_forecast = 122.96428984570457
        user_price = 266.50
        quantity = 20.0

        dual_prices = estimate_dual_scenario_input_price(
            current_wpi=base_wpi,
            forecast_wpi=unrounded_ridge_forecast,
            current_user_price=user_price,
            price_unit="₹ per 50 kg bag",
            commodity_code=1310020001,
            commodity_name="Urea",
            current_period="2025-26",
            forecast_period="2026-27",
        )

        ml_price = dual_prices["ml_trend"]
        sb_price = dual_prices["static_baseline"]

        # Exact ratio: 122.96428984570457 / 115.60 = 1.0637049294611122
        # Exact product: 266.50 * 1.0637049294611122 = 283.4773637013864
        # Standard 2-decimal rounding: round(283.4773637..., 2) = 283.48 (NOT 283.49)
        assert ml_price.estimated_future_price == 283.48
        assert sb_price.estimated_future_price == 266.50

        dual_exp = calculate_dual_scenario_expenditure(
            ml_trend_projection=ml_price,
            static_baseline_projection=sb_price,
            total_quantity=quantity,
            quantity_unit="50 kg bag",
        )

        ml_exp = dual_exp["ml_trend"]
        sb_exp = dual_exp["static_baseline"]

        # Total expenditure for 20 bags
        # Baseline: 20 * 266.50 = 5330.00
        # Scenario A (Ridge): 20 * 283.48 = 5669.60
        # Scenario B (Static): 20 * 266.50 = 5330.00
        # Variance: 5669.60 - 5330.00 = +339.60 (+6.37%)
        assert ml_exp.current_total_cost == 5330.00
        assert ml_exp.estimated_future_total_cost == 5669.60
        assert ml_exp.estimated_cost_increase == 339.60
        assert sb_exp.estimated_future_total_cost == 5330.00
        assert sb_exp.estimated_cost_increase == 0.00
