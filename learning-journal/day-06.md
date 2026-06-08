# Day 06 — Learning Journal

**Date :** January 2025
**Topic :** Multi-Table Merge + Missingness Analysis (NYC TLC Dataset)
**Dataset :** Green Taxi (68K rows) + Yellow Taxi (3M rows), January 2023

---

## What I built today

### Standard task (`day06_tlc_merge.py`)

- Loaded two Parquet files with different schemas (lpep*\* vs tpep*\* datetime columns)
- Renamed mismatched columns to shared names before merging
- Used `pd.concat` and verified row count with `assert` — no silent data loss
- Engineered 5 new features: trip_duration_min, pickup_hour, pickup_dow, speed_mph, tip_rate
- Removed physically impossible rows (negative duration, speed > 150 mph, negative fares)
- Answered 5 business questions with saved plots
- Saved a clean combined Parquet to `data/` for reuse in the challenge

### Out-of-box challenge (`day06_missingness.py`)

- Built a binary missingness matrix: `.isnull().astype(int)` on null columns
- Named flags with `MISS_` prefix for clarity
- Computed Pearson correlation (point-biserial for binary flags) against numeric predictors
- Plotted a missingness correlation heatmap
- Classified each null column as MCAR, MAR (weak), or MAR (strong)
- Applied domain reasoning to identify MNAR candidates

---

## The out-of-box challenge result

Key finding: `passenger_count`, `RatecodeID`, `congestion_surcharge`, and `store_and_fwd_flag` are all missing for the exact same 6.3% of green rows. That perfect cluster is not random — it is a vendor-level reporting failure. Vendor 1 (VeriFone) does not submit those fields for certain trip records. Classification: MAR on VendorID.

`payment_type` is a MNAR candidate. Cash payments may not generate a digital record at all, which would mean the missingness is caused by the value itself (cash). This cannot be confirmed from the data — it requires domain expert review.

`tip_rate` and `speed_mph` are structural NaNs I introduced in feature engineering. They are expected and correct, not data quality problems.

---

## What surprised me

- `ehail_fee` was 100% null in the green dataset. A whole column with no values. It still ships in the official TLC schema because the field exists for dispatch trips — but green taxis in January 2023 had none. The right move is to drop it immediately.
- The missingness cluster pattern (identical 6.3% across four columns) was immediately obvious once I looked at the null breakdown by taxi_type. I would never have seen that by just looking at overall null counts.
- Yellow taxi (3M rows) completely dominates the combined dataset. Green is 2.2% of the data. This matters enormously for any downstream model — it needs to be accounted for with class weights or stratified sampling.
- Negative fare_amount values exist in the raw data (min = -900). These are refunds or corrections, not measurement errors. I filtered them, but a more rigorous analysis would separate refunds from errors rather than dropping both.

---

## What I don't fully understand yet

- How to statistically test for MNAR. MCAR and MAR can be detected via correlation (Little's MCAR test is the formal version). MNAR has no equivalent test because the missing value is unobservable. I understand this conceptually but I have not implemented Little's test yet.
- The exact threshold for classifying MAR vs MCAR. I used |r| < 0.05 for MCAR. Different papers use different thresholds. The choice affects which imputation strategy you apply.
- Why some `trip_distance` values are enormous (max = 258,928 miles). These are clearly data errors, but I only filtered on speed and duration — not distance directly. I should add a distance cap in the cleaning step.

---

## GitHub commit made: ✅

`day-06: nyc-tlc multi-table merge + missingness MCAR/MAR/MNAR analysis`

---

## Tomorrow's priority

Day 07 — Visualization that Tells Stories.
Every chart title must be a conclusion, not a description.
Continue on the TLC dataset — it has enough patterns to build a full visual story.
