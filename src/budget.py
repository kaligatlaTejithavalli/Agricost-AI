"""Budget estimation and index-to-rupee bridge module for AgriCost-AI.

Resolves the difference between Wholesale Price Index (WPI) values and actual
rupee prices paid by farmers. Strictly enforces:
1. Target variable is 'WPI index points', never rupees.
2. The index ratio formula bridges index movements to user-supplied current rupee costs:
   estimated_future_price = current_user_price * (forecast_wpi / current_wpi)
   projected_percentage_change = ((forecast_wpi - current_wpi) / current_wpi) * 100
3. Rigorous validation against invalid inputs, non-numeric values, negative/zero WPI,
   and mismatched commodity series.
4. Transparent labeling, explanations, and disclaimers.
5. The budget planner never multiplies WPI points by farm size.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional

from src.config import (
    ASSUMPTIONS_AND_LIMITATIONS_DISCLAIMER,
    LABEL_ESTIMATED_FUTURE_PRICE,
    LABEL_ESTIMATED_PCT_CHANGE,
    LABEL_FORECAST_WPI,
    LABEL_HISTORICAL_WPI,
    LABEL_USER_CURRENT_PRICE,
    METHODOLOGY_EXPLANATION,
    TARGET_LABEL,
)


def extract_physical_unit(unit_str: str) -> str:
    """Extract clean physical quantity unit from a unit or price-unit string.

    Strips currency symbols (₹, Rs., INR), rate prepositions ('per', '/'),
    and parenthetical hiring/cultivation rate qualifiers.

    Examples:
        '₹ per 50 kg bag' -> '50 kg bag'
        '₹ per litre' -> 'litre'
        '₹ per kWh' -> 'kWh'
        '₹ per hour (hiring rate)' -> 'hour'
        '₹ per acre (harvest hiring rate)' -> 'acre'
        '₹ per quintal (threshing rate)' -> 'quintal'
        '₹ per acre (cultivation rate)' -> 'acre'
        '₹ per tyre' -> 'tyre'
        '50 kg bag' -> '50 kg bag'
        'bags' -> 'bags'
    """
    if not isinstance(unit_str, str):
        return str(unit_str).strip()
    s = unit_str.strip()
    if not s:
        return ""
    # Repeatedly remove leading currency and 'per' or '/' prefixes to prevent duplication
    while True:
        cleaned = re.sub(r"^(?:₹|Rs\.?|INR)?\s*(?:per|/)\s*", "", s, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(?:₹|Rs\.?|INR)\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == s:
            break
        s = cleaned
    # Strip parenthetical rate qualifiers (e.g., '(hiring rate)', '(harvest hiring rate)')
    s = re.sub(r"\s*\([^)]*rate[^)]*\)", "", s, flags=re.IGNORECASE).strip()
    return s if s else unit_str.strip()


def format_price_unit(unit_str: str) -> str:
    """Format a consistent currency-denominated price unit string without duplication.

    Idempotent: prevents duplicated '₹ per ₹ per ...' prefixes.

    Examples:
        '50 kg bag' -> '₹ per 50 kg bag'
        '₹ per 50 kg bag' -> '₹ per 50 kg bag'
        'litre' -> '₹ per litre'
        '₹/litre' -> '₹ per litre'
        'kWh' -> '₹ per kWh'
    """
    if not isinstance(unit_str, str):
        return f"₹ per {unit_str}"
    phys = extract_physical_unit(unit_str)
    return f"₹ per {phys}"


def format_quantity_unit(quantity: float, unit_str: str) -> str:
    """Format quantity with physical unit, applying natural pluralization.

    Examples:
        (1.0, '50 kg bag') -> '1 50 kg bag'
        (20.0, '50 kg bag') -> '20 bags'
        (20.0, 'bags') -> '20 bags'
        (10.0, 'litre') -> '10 litres'
        (1.0, 'litre') -> '1 litre'
        (5.0, 'kWh') -> '5 kWh'
    """
    phys = extract_physical_unit(unit_str)
    qty_val = float(quantity)
    qty_str = f"{int(qty_val)}" if qty_val.is_integer() else f"{qty_val:.2f}"

    if abs(qty_val - 1.0) < 1e-4:
        return f"{qty_str} {phys}"

    # Pluralization for multi-unit quantities
    if phys == "50 kg bag":
        return f"{qty_str} bags"
    elif phys.endswith("bag") and not phys.endswith("bags"):
        return f"{qty_str} {phys}s"
    elif phys in ("litre", "quintal", "tyre", "acre", "hour"):
        return f"{qty_str} {phys}s"
    elif phys.endswith("s"):
        return f"{qty_str} {phys}"
    else:
        return f"{qty_str} {phys}"


@dataclass(frozen=True)
class PriceProjectionResult:
    """Encapsulates the result of bridging a forecasted WPI movement to a rupee price.

    Maintains complete provenance and transparent labeling.
    """

    commodity_code: Optional[int]
    commodity_name: str
    series_type: str
    current_period: str
    forecast_period: str
    current_wpi: float
    forecast_wpi: float
    current_user_price: float
    price_unit: str
    estimated_future_price: float
    projected_percentage_change: float
    target_unit: str = TARGET_LABEL
    is_estimate: bool = True
    methodology_explanation: str = METHODOLOGY_EXPLANATION
    assumptions_disclaimer: str = ASSUMPTIONS_AND_LIMITATIONS_DISCLAIMER
    labels: Dict[str, str] = field(
        default_factory=lambda: {
            "historical_wpi": LABEL_HISTORICAL_WPI,
            "forecast_wpi": LABEL_FORECAST_WPI,
            "user_current_price": LABEL_USER_CURRENT_PRICE,
            "estimated_future_price": LABEL_ESTIMATED_FUTURE_PRICE,
            "projected_percentage_change": LABEL_ESTIMATED_PCT_CHANGE,
        }
    )

    @property
    def physical_unit(self) -> str:
        """Physical purchase unit extracted from price_unit (e.g. '50 kg bag', 'litre')."""
        return extract_physical_unit(self.price_unit)

    def to_dict(self) -> Dict[str, Any]:
        """Convert projection result to dictionary format for display or serialization."""
        return {
            "commodity_code": self.commodity_code,
            "commodity_name": self.commodity_name,
            "series_type": self.series_type,
            "current_period": self.current_period,
            "forecast_period": self.forecast_period,
            "target_unit": self.target_unit,
            "current_wpi_index_points": self.current_wpi,
            "forecast_wpi_index_points": self.forecast_wpi,
            "current_user_price_inr": self.current_user_price,
            "price_unit": self.price_unit,
            "physical_unit": self.physical_unit,
            "estimated_future_price_inr": self.estimated_future_price,
            "projected_percentage_change": self.projected_percentage_change,
            "is_estimate": self.is_estimate,
            "methodology_explanation": self.methodology_explanation,
            "assumptions_disclaimer": self.assumptions_disclaimer,
            "labels": self.labels,
        }


@dataclass(frozen=True)
class FarmExpenditureItem:
    """Represents an itemized input expenditure in the Farm Budget Planner.

    Ensures that input expenditure is calculated solely from valid user-entered quantities
    multiplied by the estimated future unit price (in ₹), NEVER by multiplying a WPI value.
    """

    projection: PriceProjectionResult
    quantity_per_acre: Optional[float]
    farm_size_acres: Optional[float]
    total_quantity: float
    quantity_unit: str
    current_total_cost: float
    estimated_future_total_cost: float
    estimated_cost_increase: float
    projected_percentage_change: float


@dataclass(frozen=True)
class FarmBudgetPlan:
    """Aggregate farm budget plan combining multiple itemized input expenditures."""

    items: List[FarmExpenditureItem]
    farm_size_acres: Optional[float]
    current_total_budget: float
    estimated_future_total_budget: float
    projected_net_change: float
    overall_percentage_change: float
    disclaimer: str = ASSUMPTIONS_AND_LIMITATIONS_DISCLAIMER


def validate_numeric_positive(value: Any, name: str) -> float:
    """Validate that an input value is a valid, finite, positive real number (> 0).

    Args:
        value: The value to validate.
        name: The human-readable name of the field for error reporting.

    Returns:
        float: The validated float value.

    Raises:
        ValueError: If value is None, non-numeric, NaN, infinite, zero, or negative.
    """
    if value is None:
        raise ValueError(f"Invalid input: '{name}' cannot be None or missing.")

    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input: '{name}' must be a numeric value, got '{value}'.")

    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Invalid input: '{name}' cannot be NaN or infinite.")

    if val <= 0.0:
        raise ValueError(
            f"Invalid input: '{name}' must be strictly positive (> 0), got {val}."
        )

    return val


def estimate_future_input_price(
    current_wpi: Any,
    forecast_wpi: Any,
    current_user_price: Any,
    price_unit: str,
    commodity_code: Optional[int] = None,
    commodity_name: str = "Selected Agricultural Input",
    current_period: str = "Current Observed",
    forecast_period: str = "Forecast Period",
    series_type: str = "FY",
    forecast_commodity_code: Optional[int] = None,
    forecast_series_type: Optional[str] = None,
) -> PriceProjectionResult:
    """Estimate a future agricultural input price in rupees using the WPI index-ratio formula.

    Inputs:
        - current_wpi: Current observed WPI for the selected commodity.
        - forecast_wpi: Forecast WPI for the same commodity.
        - current_user_price: Current rupee price supplied by the user.
        - price_unit: The physical unit associated with the user-provided price (e.g., '₹ per 50 kg bag').

    Formula:
        estimated_future_price = current_user_price * (forecast_wpi / current_wpi)
        projected_percentage_change = ((forecast_wpi - current_wpi) / current_wpi) * 100

    Validation:
        - Rejects missing, nonnumeric, zero, or negative WPI values where the ratio is undefined or invalid.
        - Rejects invalid current-price inputs (<= 0 or non-numeric).
        - Rejects mismatched commodity codes (never substitutes an unrelated commodity's index).
        - Rejects mismatched series types (e.g., comparing FY with CY).
        - Enforces non-empty physical unit strings.

    Returns:
        PriceProjectionResult: Object containing the estimated price, percentage change,
        labels, and clear methodology explanations and disclaimers.
    """
    # 1. Validate numeric inputs
    c_wpi = validate_numeric_positive(current_wpi, "current_wpi (observed WPI index points)")
    f_wpi = validate_numeric_positive(forecast_wpi, "forecast_wpi (forecasted WPI index points)")
    u_price = validate_numeric_positive(current_user_price, "current_user_price (rupees)")

    # 2. Validate unit string
    if not isinstance(price_unit, str) or not price_unit.strip():
        raise ValueError("Invalid physical unit: 'price_unit' must be a non-empty string (e.g., '₹ per 50 kg bag').")
    clean_unit = format_price_unit(price_unit.strip())

    # 3. Validate commodity consistency (never use an unrelated commodity's WPI)
    if commodity_code is not None and forecast_commodity_code is not None:
        if int(commodity_code) != int(forecast_commodity_code):
            raise ValueError(
                f"Commodity series mismatch: Current WPI commodity code ({commodity_code}) does not "
                f"match Forecast WPI commodity code ({forecast_commodity_code}). "
                "Never substitute an unrelated commodity's index."
            )

    # 4. Validate series compatibility (do not mix Financial Year and Calendar Year series)
    norm_series_type = str(series_type).strip().upper()
    if forecast_series_type is not None:
        norm_forecast_series = str(forecast_series_type).strip().upper()
        if norm_series_type != norm_forecast_series:
            raise ValueError(
                f"Series type mismatch: Current series '{norm_series_type}' is incompatible with "
                f"forecast series '{norm_forecast_series}'. Must compare identical series types."
            )

    # 5. Core mathematical ratio calculation
    index_ratio = f_wpi / c_wpi
    estimated_future_price = round(u_price * index_ratio, 2)
    projected_percentage_change = round(((f_wpi - c_wpi) / c_wpi) * 100.0, 2)

    return PriceProjectionResult(
        commodity_code=int(commodity_code) if commodity_code is not None else None,
        commodity_name=commodity_name,
        series_type=norm_series_type,
        current_period=current_period,
        forecast_period=forecast_period,
        current_wpi=round(c_wpi, 2),
        forecast_wpi=round(f_wpi, 2),
        current_user_price=round(u_price, 2),
        price_unit=clean_unit,
        estimated_future_price=estimated_future_price,
        projected_percentage_change=projected_percentage_change,
    )


def calculate_input_expenditure(
    price_projection: PriceProjectionResult,
    total_quantity: Optional[Any] = None,
    quantity_per_acre: Optional[Any] = None,
    farm_size_acres: Optional[Any] = None,
    quantity_unit: Optional[str] = None,
) -> FarmExpenditureItem:
    """Calculate the estimated total farm expenditure for an agricultural input.

    CRITICAL ARCHITECTURAL RULE:
    The budget planner must NEVER multiply a WPI index value directly by farm size.
    Expenditure is strictly computed from user-defined quantities and the estimated future
    rupee unit price:
        total_quantity = quantity_per_acre * farm_size_acres  (or direct total_quantity)
        estimated_future_total_cost = estimated_future_price * total_quantity

    Args:
        price_projection: Validated PriceProjectionResult from estimate_future_input_price.
        total_quantity: Direct total quantity of units required across the farm.
        quantity_per_acre: Required quantity of units per acre.
        farm_size_acres: Farm operational landholding in acres.
        quantity_unit: Physical unit label for the quantity (e.g., 'bags', 'litres', 'hours').

    Returns:
        FarmExpenditureItem: Structured itemized budget record.

    Raises:
        ValueError: If quantities are missing, non-positive, or contradictory.
    """
    if not isinstance(price_projection, PriceProjectionResult):
        raise TypeError("price_projection must be an instance of PriceProjectionResult.")

    resolved_q_per_acre: Optional[float] = None
    resolved_acres: Optional[float] = None

    if total_quantity is not None:
        qty = validate_numeric_positive(total_quantity, "total_quantity")
        if farm_size_acres is not None:
            resolved_acres = validate_numeric_positive(farm_size_acres, "farm_size_acres")
            resolved_q_per_acre = round(qty / resolved_acres, 4)
    elif quantity_per_acre is not None and farm_size_acres is not None:
        resolved_q_per_acre = validate_numeric_positive(quantity_per_acre, "quantity_per_acre")
        resolved_acres = validate_numeric_positive(farm_size_acres, "farm_size_acres")
        qty = resolved_q_per_acre * resolved_acres
    else:
        raise ValueError(
            "Missing usage quantity: You must provide either an explicit 'total_quantity', "
            "or both 'quantity_per_acre' and 'farm_size_acres'. Do not invent crop usage rates."
        )

    # Resolve unit string as clean physical quantity unit
    if quantity_unit and quantity_unit.strip():
        unit_label = extract_physical_unit(quantity_unit)
    else:
        unit_label = price_projection.physical_unit

    current_total = round(price_projection.current_user_price * qty, 2)
    future_total = round(price_projection.estimated_future_price * qty, 2)
    cost_increase = round(future_total - current_total, 2)

    return FarmExpenditureItem(
        projection=price_projection,
        quantity_per_acre=resolved_q_per_acre,
        farm_size_acres=resolved_acres,
        total_quantity=round(qty, 2),
        quantity_unit=unit_label,
        current_total_cost=current_total,
        estimated_future_total_cost=future_total,
        estimated_cost_increase=cost_increase,
        projected_percentage_change=price_projection.projected_percentage_change,
    )


def calculate_farm_budget(
    items: List[FarmExpenditureItem],
    farm_size_acres: Optional[float] = None,
) -> FarmBudgetPlan:
    """Aggregate multiple input expenditure items into a complete farm budget plan.

    Args:
        items: List of FarmExpenditureItem objects.
        farm_size_acres: Optional operational acreage.

    Returns:
        FarmBudgetPlan: Consolidated budget plan with transparent baseline vs projected comparisons.

    Raises:
        ValueError: If items list is empty.
    """
    if not items:
        raise ValueError("Budget plan requires at least one expenditure item.")

    current_total = sum(item.current_total_cost for item in items)
    future_total = sum(item.estimated_future_total_cost for item in items)
    net_change = round(future_total - current_total, 2)

    pct_change = round(((future_total - current_total) / current_total) * 100.0, 2) if current_total > 0 else 0.0

    return FarmBudgetPlan(
        items=items,
        farm_size_acres=farm_size_acres,
        current_total_budget=round(current_total, 2),
        estimated_future_total_budget=round(future_total, 2),
        projected_net_change=net_change,
        overall_percentage_change=pct_change,
    )


def estimate_dual_scenario_input_price(
    current_wpi: Any,
    forecast_wpi: Any,
    current_user_price: Any,
    price_unit: str,
    commodity_code: Optional[int] = None,
    commodity_name: str = "Selected Agricultural Input",
    current_period: str = "Current Observed",
    forecast_period: str = "Forecast Period",
    series_type: str = "FY",
    forecast_commodity_code: Optional[int] = None,
    forecast_series_type: Optional[str] = None,
) -> Dict[str, PriceProjectionResult]:
    """Estimate future input price for both ML Trend and Static Baseline scenarios.

    Formula:
        ML Trend Future Price: current_user_price * (forecast_wpi / current_wpi)
        Static Baseline Future Price: current_user_price * (current_wpi / current_wpi) = current_user_price

    Returns:
        Dict[str, PriceProjectionResult] containing:
            'ml_trend': PriceProjectionResult using forecasted WPI.
            'static_baseline': PriceProjectionResult using static baseline (forecast_wpi = current_wpi).
    """
    ml_trend = estimate_future_input_price(
        current_wpi=current_wpi,
        forecast_wpi=forecast_wpi,
        current_user_price=current_user_price,
        price_unit=price_unit,
        commodity_code=commodity_code,
        commodity_name=commodity_name,
        current_period=current_period,
        forecast_period=forecast_period,
        series_type=series_type,
        forecast_commodity_code=forecast_commodity_code,
        forecast_series_type=forecast_series_type,
    )

    static_baseline = estimate_future_input_price(
        current_wpi=current_wpi,
        forecast_wpi=current_wpi,  # Flat baseline: zero index movement
        current_user_price=current_user_price,
        price_unit=price_unit,
        commodity_code=commodity_code,
        commodity_name=commodity_name,
        current_period=current_period,
        forecast_period=forecast_period,
        series_type=series_type,
        forecast_commodity_code=commodity_code,
        forecast_series_type=series_type,
    )

    return {
        "ml_trend": ml_trend,
        "static_baseline": static_baseline,
    }


def calculate_dual_scenario_expenditure(
    ml_trend_projection: PriceProjectionResult,
    static_baseline_projection: PriceProjectionResult,
    total_quantity: Optional[Any] = None,
    quantity_per_acre: Optional[Any] = None,
    farm_size_acres: Optional[Any] = None,
    quantity_unit: Optional[str] = None,
) -> Dict[str, FarmExpenditureItem]:
    """Calculate farm expenditure for both ML Trend and Static Baseline scenarios.

    Returns:
        Dict[str, FarmExpenditureItem] containing:
            'ml_trend': FarmExpenditureItem under ML Trend scenario.
            'static_baseline': FarmExpenditureItem under Static Baseline scenario.
    """
    ml_item = calculate_input_expenditure(
        price_projection=ml_trend_projection,
        total_quantity=total_quantity,
        quantity_per_acre=quantity_per_acre,
        farm_size_acres=farm_size_acres,
        quantity_unit=quantity_unit,
    )
    static_item = calculate_input_expenditure(
        price_projection=static_baseline_projection,
        total_quantity=total_quantity,
        quantity_per_acre=quantity_per_acre,
        farm_size_acres=farm_size_acres,
        quantity_unit=quantity_unit,
    )
    return {
        "ml_trend": ml_item,
        "static_baseline": static_item,
    }

