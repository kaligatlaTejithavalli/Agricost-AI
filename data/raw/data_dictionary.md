# AgriCost-AI — Data Dictionary & Provenance Audit
**Dataset:** `INDIAN_GOVT_WHOLESALE_PRICE_INDEX.csv`  
**Governing Institution:** Office of the Economic Adviser (OEA), Department for Promotion of Industry and Internal Trade (DPIIT), Ministry of Commerce and Industry, Government of India.  
**Official Series:** Wholesale Price Index (WPI), Base Year 2011-12 = 100.

---

## 1. Dataset Overview & Provenance

The Wholesale Price Index (WPI) is a national macroeconomic price benchmark published monthly by the Government of India to measure average price changes in goods sold and traded in bulk at the wholesale stage.

- **Primary Source:** Office of the Economic Adviser (OEA), DPIIT, Ministry of Commerce and Industry.
- **Base Year:** 2011-12 = 100.0 (Index base established in 2017 revision).
- **Physical Dataset Path:** `data/raw/INDIAN_GOVT_WHOLESALE_PRICE_INDEX.csv`
- **Total Rows:** 869 rows (comprising major groups, subgroups, item groups, and individual item commodities).
- **Total Columns:** 27 columns (3 metadata columns + 24 index period columns).
- **Measurement Unit of Index Columns:** **WPI index points** (dimensionless statistical index normalized to base 2011-12 = 100.0).
- **CRITICAL DISTINCTION:** WPI values are **not** prices in Indian Rupees (₹). An index value of `164.0` represents a 64% increase relative to the 2011-12 base wholesale benchmark; it does **not** equal ₹164.00, ₹/bag, ₹/kg, or ₹/litre.

---

## 2. Column Schema & Data Types

| Column Name | Data Type | Description | Observed Range / Values | Missing Count |
| :--- | :--- | :--- | :--- | :--- |
| `COMM_NAME` | `string` | Commodity or Group Name as designated by DPIIT | 869 unique labels | 0 (0.0%) |
| `COMM_CODE` | `int64` / `string` | 10-digit hierarchical classification code | `1000000000` to `2000000000` | 0 (0.0%) |
| `COMM_WT` | `float64` | Weight in the All-Commodities WPI basket (out of 100.0) | `0.00071` to `100.0` | 0 (0.0%) |
| `INDEX2011` | `float64` / `NA` | Annual wholesale price index for Calendar Year 2011 | `NA` | 869 (100.0%) |
| `INDEX20112012` | `float64` / `NA` | Wholesale price index for Financial Year 2011-12 (Base = 100) | `NA` | 869 (100.0%) |
| `INDEX2012` | `float64` / `NA` | Annual wholesale price index for Calendar Year 2012 | `NA` | 869 (100.0%) |
| `INDEX20122013` | `float64` | Financial Year 2012-13 average index (Apr 2012 – Mar 2013) | `38.3` to `186.8` (mean: 106.35) | 0 (0.0%) |
| `INDEX2013` | `float64` | Calendar Year 2013 average index (Jan 2013 – Dec 2013) | `59.2` to `282.3` (mean: 110.65) | 0 (0.0%) |
| `INDEX20132014` | `float64` | Financial Year 2013-14 average index (Apr 2013 – Mar 2014) | `63.3` to `280.2` (mean: 111.99) | 0 (0.0%) |
| `INDEX2014` | `float64` | Calendar Year 2014 average index (Jan 2014 – Dec 2014) | `60.0` to `285.0` (mean: 115.30) | 0 (0.0%) |
| `INDEX20142015` | `float64` | Financial Year 2014-15 average index (Apr 2014 – Mar 2015) | `59.7` to `266.2` (mean: 115.66) | 0 (0.0%) |
| `INDEX2015` | `float64` | Calendar Year 2015 average index (Jan 2015 – Dec 2015) | `57.4` to `263.1` (mean: 115.27) | 0 (0.0%) |
| `INDEX20152016` | `float64` | Financial Year 2015-16 average index (Apr 2015 – Mar 2016) | `54.2` to `255.5` (mean: 115.21) | 0 (0.0%) |
| `INDEX2016` | `float64` | Calendar Year 2016 average index (Jan 2016 – Dec 2016) | `48.5` to `261.9` (mean: 116.32) | 0 (0.0%) |
| `INDEX20162017` | `float64` | Financial Year 2016-17 average index (Apr 2016 – Mar 2017) | `48.3` to `248.0` (mean: 117.05) | 0 (0.0%) |
| `INDEX2017` | `float64` | Calendar Year 2017 average index (Jan 2017 – Dec 2017) | `46.7` to `242.4` (mean: 118.60) | 0 (0.0%) |
| `INDEX20172018` | `float64` | Financial Year 2017-18 average index (Apr 2017 – Mar 2018) | `46.7` to `236.5` (mean: 119.20) | 0 (0.0%) |
| `INDEX2018` | `float64` | Calendar Year 2018 average index (Jan 2018 – Dec 2018) | `40.9` to `288.3` (mean: 121.81) | 0 (0.0%) |
| `INDEX20182019` | `float64` | Financial Year 2018-19 average index (Apr 2018 – Mar 2019) | `30.9` to `311.2` (mean: 122.76) | 0 (0.0%) |
| `INDEX2019` | `float64` | Calendar Year 2019 average index (Jan 2019 – Dec 2019) | `44.3` to `337.9` (mean: 125.34) | 0 (0.0%) |
| `INDEX20192020` | `float64` | Financial Year 2019-20 average index (Apr 2019 – Mar 2020) | `43.6` to `350.8` (mean: 125.91) | 0 (0.0%) |
| `INDEX2020` | `float64` | Calendar Year 2020 average index (Jan 2020 – Dec 2020) | `41.8` to `400.6` (mean: 126.05) | 0 (0.0%) |
| `INDEX20202021` | `float64` | Financial Year 2020-21 average index (Apr 2020 – Mar 2021) | `42.2` to `371.6` (mean: 127.24) | 0 (0.0%) |
| `INDEX2021` | `float64` | Calendar Year 2021 average index (Jan 2021 – Dec 2021) | `44.3` to `327.0` (mean: 136.33) | 0 (0.0%) |
| `INDEX20212022` | `float64` | Financial Year 2021-22 average index (Apr 2021 – Mar 2022) | `44.8` to `372.2` (mean: 140.17) | 0 (0.0%) |
| `INDEX2022` | `float64` | Calendar Year 2022 average index (Jan 2022 – Dec 2022) | `47.6` to `413.5` (mean: 148.90) | 0 (0.0%) |
| `INDEX20222023` | `float64` | Financial Year 2022-23 average index (Apr 2022 – Mar 2023) | `48.2` to `410.3` (mean: 149.63) | 0 (0.0%) |

---

## 3. Analysis of Time Periods & Temporal Structure

A critical inspection reveals that the dataset column headers contain **two interleaved time series**:

1. **Financial Year (FY) Series (April 1 to March 31):**
   - Columns: `INDEX20122013`, `INDEX20132014`, `INDEX20142015`, `INDEX20152016`, `INDEX20162017`, `INDEX20172018`, `INDEX20182019`, `INDEX20192020`, `INDEX20202021`, `INDEX20212022`, `INDEX20222023` (11 annual periods).
   - *Academic Significance:* The Financial Year is the official accounting and agricultural budget period in India (coinciding with the union budget, fertilizer subsidy allocation, and CACP agricultural cost reporting).
2. **Calendar Year (CY) Series (January 1 to December 31):**
   - Columns: `INDEX2013`, `INDEX2014`, `INDEX2015`, `INDEX2016`, `INDEX2017`, `INDEX2018`, `INDEX2019`, `INDEX2020`, `INDEX2021`, `INDEX2022` (10 annual periods).
3. **Temporal Incompatibility Warning:**
   - Calendar Year 2013 (Jan–Dec 2013) and Financial Year 2013-14 (Apr 2013–Mar 2014) overlap by 9 months.
   - **Rule:** The system must never treat all 21 columns as a single 21-step chronological sequence. Either the Financial Year series or the Calendar Year series must be selected consistently for forecasting and index-ratio calculations. Financial Year is the default recommended series for Indian agricultural planning.
4. **Missing Values Audit:**
   - `INDEX2011`, `INDEX20112012`, and `INDEX2012` are 100% missing (`NA` string in the raw file).
   - In accordance with the project rules, these observations are **not** imputed or fabricated. Valid time series begin from `INDEX20122013` for Financial Years and `INDEX2013` for Calendar Years.

---

## 4. Commodity Classification: Agricultural Inputs vs. Outputs

To avoid erroneous assumptions, the 869 commodities in the dataset have been audited and categorized:

### 4.1 Plausible Agricultural Input Commodities (Included in AgriCost-AI)

These commodities represent actual materials, energy, or equipment purchased by farmers as inputs to agricultural operations:

| Commodity Code | Commodity Name | WPI Weight | Input Category | Typical Farm Use / Physical Rupee Unit |
| :--- | :--- | :--- | :--- | :--- |
| `1310020001` | **Urea** | 0.72685 | Fertilizer | Nitrogenous crop nutrition (typically sold in ₹ per 45kg or 50kg bag) |
| `1310020002` | **Di ammonium phosphate** | 0.07425 | Fertilizer | Phosphatic & nitrogenous basal fertilizer (typically ₹ per 50kg bag) |
| `1310020005` | **Mixed fertilizer** | 0.45476 | Fertilizer | NPK complex fertilizer grades (typically ₹ per 50kg bag) |
| `1310020009` | **Superphospate/Phosphatic fertilizer, others** | 0.08472 | Fertilizer | Single Super Phosphate [SSP] (typically ₹ per 50kg bag) |
| `1310020006` | **Ammonium phosphate** | 0.04093 | Fertilizer | Complex fertilizer (typically ₹ per 50kg bag) |
| `1310020003` | **Ammonium sulphate** | 0.03420 | Fertilizer | Nitrogenous & sulfur fertilizer (typically ₹ per 50kg bag) |
| `1310020004` | **Nitrogenous fertilizer, others** | 0.00427 | Fertilizer | Specialized nitrogen fertilizers (typically ₹ per 50kg bag) |
| `1310020000` | **b. Manufacture of fertilizers and nitrogen compounds** | 1.48497 | Fertilizer (Group) | Broad fertilizer sector aggregate index |
| `1310040001` | **Insecticide and Pesticide** | 0.38578 | Agrochemical | Crop protection chemicals (typically ₹ per litre or ₹ per kg) |
| `1310040002` | **Agro chemical formulation** | 0.06434 | Agrochemical | Technical agrochemical blends (typically ₹ per litre or ₹ per kg) |
| `1310040003` | **Fungicide, liquid** | 0.00357 | Agrochemical | Fungal disease treatment (typically ₹ per litre) |
| `1310040000` | **d. Manufacture of pesticides and other agrochemical products** | 0.45369 | Agrochemical (Group) | Broad agrochemical sector aggregate index |
| `1202000005` | **HSD** (High Speed Diesel) | 3.09548 | Fuel & Power | Tractor operation, diesel pump irrigation, harvesting (₹ per litre) |
| `1202000002` | **Petrol** | 1.60470 | Fuel & Power | Small power weeders, sprayers, farm utility transport (₹ per litre) |
| `1203000001` | **Electricity** | 3.06409 | Fuel & Power | Tube-well and pump set irrigation power (₹ per kWh / unit) |
| `1318090001` | **Agricultural tractors** | 0.52772 | Farm Machinery | Primary farm mechanization (custom hiring rate: ₹ per hour / acre) |
| `1318090003` | **Agriculture implements** | 0.26218 | Farm Machinery | Tillers, harrows, seed drills (₹ per unit or hiring rate) |
| `1318090002` | **Harvesters** | 0.00364 | Farm Machinery | Combine harvesting services (custom hiring rate: ₹ per hour / acre) |
| `1318090004` | **Threshers** | 0.00071 | Farm Machinery | Post-harvest crop threshing (₹ per quintal or hiring rate) |
| `1318090005` | **Soil preparation & cultivation machinery (other than tractors)** | 0.03840 | Farm Machinery | Rotavators, plows, disc harrows |
| `1312010002` | **Tractor Tyre** | 0.03327 | Farm Machinery | Maintenance consumable for farm tractors (₹ per tyre) |
| `1318090000` | **i. Manufacture of agricultural and forestry machinery** | 0.83265 | Farm Machinery (Group) | Broad farm equipment aggregate index |
| `1102030008` | **Fodder** | 0.53140 | Livestock Input | Green and dry cattle fodder (₹ per quintal) |
| `1301160000` | **p. Manufacture of prepared animal feeds** | 0.35630 | Livestock Input | Cattle and poultry feed compound (₹ per 50kg bag) |
| `1301160004` | **Cotton seed oil cake** | 0.06115 | Livestock Input | Dairy animal protein feed supplement (₹ per quintal / bag) |
| `1301160005` | **Mustard oil cake** | 0.03512 | Livestock Input | Cattle feed supplement (₹ per quintal / bag) |

### 4.2 Excluded Commodities (Outputs, Not Inputs)

The following items appear in the WPI dataset but are **strictly excluded** from being treated as agricultural inputs:

1. **Crop Harvests / Primary Food Articles (COMM_CODE: 1101...):**
   - Examples: Bajra, Masur, Tomato, Cauliflower, Brinjal, Sweet Potato, Apple, Litchi, Drumstick, Turmeric, Tea, Coffee.
   - *Rationale:* These are farm outputs produced and sold by farmers to generate revenue; they are not costs incurred to cultivate crops.
2. **Oilseeds (COMM_CODE: 110202...):**
   - Examples: Groundnut Seed, Rape & Mustard Seed, Cotton Seed, Linseed, Niger Seed, Castor Seed, Safflower Seed.
   - *Rationale:* In the WPI classification, these represent commercial oilseed crops harvested by farmers and sold to crushing mills. They do not represent certified planting seeds sold by government seed corporations.
3. **Manufactured Consumer & Industrial Goods:**
   - Examples: Textiles, basic metals, paper, pharmaceuticals, electronics, furniture.
   - *Rationale:* Unrelated to agricultural crop production.

### 4.3 Inputs Missing from WPI Data (Documented Limitations)

1. **Agricultural Field Labour:**
   - Daily wage rates for casual field labour (ploughing, weeding, harvesting) are **not** tracked in WPI. Labour wages are tracked separately by the Labour Bureau through the Consumer Price Index for Agricultural Labourers (CPI-AL) and Rural Wage Rates.
2. **Certified Hybrid Seeds:**
   - Proprietary hybrid seed varieties (e.g., Bt Cotton, hybrid maize, certified paddy seeds) are not itemized in the WPI.

---

## 5. Mathematical Bridge: WPI Index to Rupee Price Estimation

Because WPI is an index and not a rupee quote, estimating future rupee prices requires combining the macroeconomic index trend with the user's observed local market price:

$$\text{Estimated Future Price} = \text{Current User Price} \times \left( \frac{\text{Forecast WPI}}{\text{Current WPI}} \right)$$

$$\text{Projected Percentage Change} = \left( \frac{\text{Forecast WPI} - \text{Current WPI}}{\text{Current WPI}} \right) \times 100$$

### Critical Assumptions & Limitations:
1. **Pass-Through Assumption:** The formula assumes that wholesale price index changes pass through proportionally (1:1 elasticity) to retail and farmgate prices. In reality, local dealer margins, transport bottlenecks, and regional subsidies may cause local retail prices to diverge from wholesale indices.
2. **Subsidized Commodities (e.g., Urea):** Urea prices in India are statutory and heavily subsidized under the Nutrient Based Subsidy (NBS) regime. As seen in the historical data, the WPI for Urea rose by only ~13% over a decade (from 101.7 to 115.1), while decontrolled inputs like DAP and Diesel experienced major market-driven fluctuations.
3. **Estimation Disclaimer:** The resulting rupee value is a planning estimate derived from index movements, **not** a guaranteed market quotation or official government procurement price.
