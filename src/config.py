"""Configuration constants, paths, and metadata schemas for AgriCost-AI.

Centralizes paths, target variable labels, agricultural commodity catalogs,
and estimation disclaimers.
"""

from pathlib import Path
from typing import Dict, List, NamedTuple

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
EVALUATION_DIR = MODELS_DIR / "evaluation"
SHAP_DIR = EVALUATION_DIR / "shap"
FORECASTS_DIR = EVALUATION_DIR / "forecasts"
REPORTS_DIR = BASE_DIR / "reports"

RAW_WPI_CSV = RAW_DATA_DIR / "INDIAN_GOVT_WHOLESALE_PRICE_INDEX.csv"
STAGED_DATA_DIR = DATA_DIR / "staged"
STAGED_WPI_XLS = STAGED_DATA_DIR / "fin_year_index.xls"
EXTENDED_PROCESSED_TIMESERIES_CSV = PROCESSED_DATA_DIR / "agricultural_inputs_wpi_fy_timeseries_extended.csv"
EXTENDED_FEATURES_CSV = PROCESSED_DATA_DIR / "agricultural_inputs_features_extended.csv"
EXTENDED_WIDE_CSV = PROCESSED_DATA_DIR / "agricultural_inputs_wpi_fy_wide_extended.csv"
EXTENDED_SHAP_DIR = EVALUATION_DIR / "shap" / "extended"
EXTENDED_FORECASTS_CSV = FORECASTS_DIR / "commodity_forecasts_extended_2026_2029.csv"


# ---------------------------------------------------------------------------
# Target Variable & Label Constants (Task 2 & Task 4)
# ---------------------------------------------------------------------------
# The forecasting target is strictly the commodity's WPI index value, NEVER rupees.
TARGET_LABEL = "WPI index points"
BASE_YEAR_LABEL = "Base Year 2011-12 = 100.0"

# Explicit, transparent labels required by Task 4
LABEL_HISTORICAL_WPI = "Historical WPI (WPI index points)"
LABEL_FORECAST_WPI = "Forecast WPI (WPI index points)"
LABEL_USER_CURRENT_PRICE = "User-Provided Current Price (₹)"
LABEL_ESTIMATED_FUTURE_PRICE = "Estimated Future Price (₹)"
LABEL_ESTIMATED_PCT_CHANGE = "Estimated Percentage Change (%)"

# Explanation strings to accompany budget & future price calculations
METHODOLOGY_EXPLANATION = (
    "Calculation Method: The estimated future price is computed by applying the projected "
    "relative movement in the official Wholesale Price Index (WPI) to the current rupee price "
    "provided by the user: "
    "estimated_future_price = current_user_price * (forecast_wpi / current_wpi)."
)

ASSUMPTIONS_AND_LIMITATIONS_DISCLAIMER = (
    "Assumptions & Limitations: "
    "1. Index Pass-Through: Assumes wholesale price index changes pass through proportionally to "
    "local retail / farmgate input prices. Local dealer margins, transport costs, and localized supply "
    "variations may cause retail prices to deviate from national wholesale index movements. "
    "2. Subsidized vs Market Prices: Government-subsidized inputs (e.g., statutory MRP for Urea) "
    "exhibit price stability dictated by policy rather than free-market index swings. "
    "3. Planning Tool Only: This figure is an analytical estimate for budget planning, "
    "NOT a guaranteed market quotation or official government-notified price."
)

# ---------------------------------------------------------------------------
# Time Series Period Definitions
# ---------------------------------------------------------------------------
# The dataset contains two interleaved series.
# Financial Years (April to March) are the primary series for Indian agricultural planning.
FY_COLUMNS: List[str] = [
    "INDEX20122013",
    "INDEX20132014",
    "INDEX20142015",
    "INDEX20152016",
    "INDEX20162017",
    "INDEX20172018",
    "INDEX20182019",
    "INDEX20192020",
    "INDEX20202021",
    "INDEX20212022",
    "INDEX20222023",
]

FY_LABELS: Dict[str, str] = {
    "INDEX20122013": "2012-13",
    "INDEX20132014": "2013-14",
    "INDEX20142015": "2014-15",
    "INDEX20152016": "2015-16",
    "INDEX20162017": "2016-17",
    "INDEX20172018": "2017-18",
    "INDEX20182019": "2018-19",
    "INDEX20192020": "2019-20",
    "INDEX20202021": "2020-21",
    "INDEX20212022": "2021-22",
    "INDEX20222023": "2022-23",
}

# Extended Financial Year series including staged observations through FY 2025-26 (Phase 3A)
EXTENDED_FY_COLUMNS: List[str] = FY_COLUMNS + [
    "INDEX20232024",
    "INDEX20242025",
    "INDEX20252026",
]

EXTENDED_FY_LABELS: Dict[str, str] = {
    **FY_LABELS,
    "INDEX20232024": "2023-24",
    "INDEX20242025": "2024-25",
    "INDEX20252026": "2025-26",
}

# Calendar Years (January to December)
CY_COLUMNS: List[str] = [
    "INDEX2013",
    "INDEX2014",
    "INDEX2015",
    "INDEX2016",
    "INDEX2017",
    "INDEX2018",
    "INDEX2019",
    "INDEX2020",
    "INDEX2021",
    "INDEX2022",
]

CY_LABELS: Dict[str, str] = {
    "INDEX2013": "2013",
    "INDEX2014": "2014",
    "INDEX2015": "2015",
    "INDEX2016": "2016",
    "INDEX2017": "2017",
    "INDEX2018": "2018",
    "INDEX2019": "2019",
    "INDEX2020": "2020",
    "INDEX2021": "2021",
    "INDEX2022": "2022",
}

# Unpopulated / Missing Columns in raw source (100% NA)
UNPOPULATED_COLUMNS: List[str] = [
    "INDEX2011",
    "INDEX20112012",
    "INDEX2012",
]

# ---------------------------------------------------------------------------
# Verified Agricultural Input Commodities Catalog
# ---------------------------------------------------------------------------
class CommodityMetadata(NamedTuple):
    code: int
    name: str
    weight: float
    category: str
    suggested_unit: str
    description: str


# Explicitly verified commodities that plausibly represent agricultural inputs.
# Excludes crop harvest outputs (fruits, vegetables, cereals) and unrelated industrial goods.
AGRICULTURAL_INPUT_CATALOG: Dict[int, CommodityMetadata] = {
    # Fertilizers
    1310020001: CommodityMetadata(
        code=1310020001,
        name="Urea",
        weight=0.72685,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="Nitrogenous basal and top-dressing crop fertilizer (statutory MRP regime)",
    ),
    1310020002: CommodityMetadata(
        code=1310020002,
        name="Di ammonium phosphate",
        weight=0.07425,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="DAP basal phosphatic fertilizer (decontrolled Nutrient Based Subsidy)",
    ),
    1310020005: CommodityMetadata(
        code=1310020005,
        name="Mixed fertilizer",
        weight=0.45476,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="NPK complex fertilizer grades",
    ),
    1310020009: CommodityMetadata(
        code=1310020009,
        name="Superphospate/Phosphatic fertilizer, others",
        weight=0.08472,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="Single Super Phosphate (SSP) phosphatic fertilizer",
    ),
    1310020006: CommodityMetadata(
        code=1310020006,
        name="Ammonium phosphate",
        weight=0.04093,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="Specialized ammonium phosphate fertilizer compound",
    ),
    1310020003: CommodityMetadata(
        code=1310020003,
        name="Ammonium sulphate",
        weight=0.03420,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="Ammonium sulphate nitrogenous and sulfur-rich fertilizer",
    ),
    1310020004: CommodityMetadata(
        code=1310020004,
        name="Nitrogenous fertilizer, others",
        weight=0.00427,
        category="Fertilizer",
        suggested_unit="₹ per 50 kg bag",
        description="Other nitrogenous fertilizer formulations",
    ),
    1310020000: CommodityMetadata(
        code=1310020000,
        name="b. Manufacture of fertilizers and nitrogen compounds",
        weight=1.48497,
        category="Fertilizer (Group Index)",
        suggested_unit="₹ per 50 kg bag",
        description="Aggregate index for all fertilizer manufacturing in India",
    ),
    # Agrochemicals / Pesticides
    1310040001: CommodityMetadata(
        code=1310040001,
        name="Insecticide and Pesticide",
        weight=0.38578,
        category="Pesticide / Agrochemical",
        suggested_unit="₹ per litre",
        description="Crop protection insecticides and general pesticides",
    ),
    1310040002: CommodityMetadata(
        code=1310040002,
        name="Agro chemical formulation",
        weight=0.06434,
        category="Pesticide / Agrochemical",
        suggested_unit="₹ per litre",
        description="Technical and ready-to-use agrochemical crop formulations",
    ),
    1310040003: CommodityMetadata(
        code=1310040003,
        name="Fungicide, liquid",
        weight=0.00357,
        category="Pesticide / Agrochemical",
        suggested_unit="₹ per litre",
        description="Liquid fungal pathogen control chemical",
    ),
    1310040000: CommodityMetadata(
        code=1310040000,
        name="d. Manufacture of pesticides and other agrochemical products",
        weight=0.45369,
        category="Pesticide / Agrochemical (Group Index)",
        suggested_unit="₹ per litre",
        description="Aggregate index for pesticide and agrochemical manufacturing",
    ),
    # Fuel & Power
    1202000005: CommodityMetadata(
        code=1202000005,
        name="HSD",
        weight=3.09548,
        category="Fuel & Power",
        suggested_unit="₹ per litre",
        description="High Speed Diesel for tractors, harvesters, and diesel pump sets",
    ),
    1202000002: CommodityMetadata(
        code=1202000002,
        name="Petrol",
        weight=1.60470,
        category="Fuel & Power",
        suggested_unit="₹ per litre",
        description="Motor spirit / petrol for small power weeders, sprayers, farm utility transport",
    ),
    1203000001: CommodityMetadata(
        code=1203000001,
        name="Electricity",
        weight=3.06409,
        category="Fuel & Power",
        suggested_unit="₹ per kWh",
        description="Agricultural electricity tariff for electric tube-well irrigation",
    ),
    # Farm Machinery & Equipment
    1318090001: CommodityMetadata(
        code=1318090001,
        name="Agricultural tractors",
        weight=0.52772,
        category="Farm Machinery",
        suggested_unit="₹ per hour (hiring rate)",
        description="Agricultural tractors; proxy for custom hiring charges per hour/acre",
    ),
    1318090003: CommodityMetadata(
        code=1318090003,
        name="Agriculture implements",
        weight=0.26218,
        category="Farm Machinery",
        suggested_unit="₹ per hour (hiring rate)",
        description="Tillage and planting implements (tillers, seed drills, disc harrows)",
    ),
    1318090002: CommodityMetadata(
        code=1318090002,
        name="Harvesters",
        weight=0.00364,
        category="Farm Machinery",
        suggested_unit="₹ per acre (harvest hiring rate)",
        description="Combine harvesters for paddy and wheat operations",
    ),
    1318090004: CommodityMetadata(
        code=1318090004,
        name="Threshers",
        weight=0.00071,
        category="Farm Machinery",
        suggested_unit="₹ per quintal (threshing rate)",
        description="Crop threshing machinery",
    ),
    1318090005: CommodityMetadata(
        code=1318090005,
        name="Soil preparation & cultivation machinery (other than tractors)",
        weight=0.03840,
        category="Farm Machinery",
        suggested_unit="₹ per acre (cultivation rate)",
        description="Rotavators, power tillers, and specialized cultivation machinery",
    ),
    1312010002: CommodityMetadata(
        code=1312010002,
        name="Tractor Tyre",
        weight=0.03327,
        category="Farm Machinery",
        suggested_unit="₹ per tyre",
        description="Tractor replacement tyres and consumables",
    ),
    1318090000: CommodityMetadata(
        code=1318090000,
        name="i. Manufacture of agricultural and forestry machinery",
        weight=0.83265,
        category="Farm Machinery (Group Index)",
        suggested_unit="₹ per hour (hiring rate)",
        description="Aggregate machinery index for agricultural and forestry equipment",
    ),
    # Livestock Inputs (Animal Feed & Fodder)
    1102030008: CommodityMetadata(
        code=1102030008,
        name="Fodder",
        weight=0.53140,
        category="Livestock Input",
        suggested_unit="₹ per quintal",
        description="Green and dry cattle fodder for livestock farming",
    ),
    1301160000: CommodityMetadata(
        code=1301160000,
        name="p. Manufacture of prepared animal feeds",
        weight=0.35630,
        category="Livestock Input",
        suggested_unit="₹ per 50 kg bag",
        description="Compound animal feeds and nutritional concentrates",
    ),
    1301160004: CommodityMetadata(
        code=1301160004,
        name="Cotton seed oil cake",
        weight=0.06115,
        category="Livestock Input",
        suggested_unit="₹ per 50 kg bag",
        description="Cotton seed oil cake livestock protein feed supplement",
    ),
    1301160005: CommodityMetadata(
        code=1301160005,
        name="Mustard oil cake",
        weight=0.03512,
        category="Livestock Input",
        suggested_unit="₹ per 50 kg bag",
        description="Mustard oil cake livestock feed supplement",
    ),
}
