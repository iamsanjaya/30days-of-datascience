# Day 06 - Standard Task
# NYC TLC Multi-Table Merge, Feature Engineering & EDA
# Dataset : Green Taxi + Yellow Taxi, January 2023
# Stack   : Pandas, NumPy, MAtplotlib, Seaborn
# Run     : VS Code # %% cells (Shift +Enter per cell)
# Output  : ../data/processed/tlc_combined_clean_jan2023.parquet
#           ../outputs/q1_trips_by_hour.png
#           ../outputs/q2_median_fare.png
#           ../outputs/q3_trip_distance.png
#           ../outputs/q4_tip_by_payment.png
#           ../outputs/q5_speed_by_hour.png


# %% Import & Paths
# All imports at the top. All paths in one place.
# If you move the project, only this cell changes.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "")


os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("[OK] Imports and paths ready.")
print(f"[LOG] Data raw: {os.path.abspath(DATA_RAW)}")
print(f"[LOG] Data processed: {os.path.abspath(DATA_PROCESSED)}")
print(f"[LOG] Outputs dir: {os.path.abspath(OUTPUTS_DIR)}")


# %% Load Both Datasets
# CONCEPT: Real-world data lives in multiple files.
# Green = borough taxi (outer NYC), Yellow = Manhattan taxi.
# We load them separately, tag their source, then caombine.
# Tagging before concat is critical - after concat you lose
# which row came from which file.

green = pd.read_parquet(DATA_RAW + "green_tripdata_2023-01.parquet")
yellow = pd.read_parquet(DATA_RAW + "yellow_tripdata_2023-01.parquet")


green["taxi_type"] = "green"
yellow["taxi_type"] = "yellow"

print(f"[RESULT] Green: {green.shape[0]:,} rows, {green.shape[1]} columns")
print(f"[RESULT] Yellow: {yellow.shape[0]:,} rows, {yellow.shape[1]} columns")
print(f"\n[LOG] Green columns: {sorted(green.columns.tolist())}")
print(f"\n[LOG] Yellow columns: {sorted(yellow.columns.tolist())}")


# %% Align Schemas Before Merging
# CONCEPT: pd.concat requires matching column names.
# Green usses lpep_* datetimes; Yellow uses tpep_* datetimes.
# We rename both to a shared neutral name.
# Columns that exist only in one dataset get dropped here -
# ehail_fee is 100% nyll in green (useless).
# airport_fee has no equivalent in green.

green_only = set(green.columns) - set(yellow.columns)
yellow_only = set(yellow.columns) - set(green.columns)

print(f"[LOG] Columns only in green: {green_only}")
print(f"[LOG] Columns only in yellow: {yellow_only}")


green = green.rename(
    columns={
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
    }
)
yellow = yellow.rename(
    columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    }
)

green = green.drop(columns=["ehail_fee", "trip_type"], errors="ignore")
yellow = yellow.drop(columns=["airport_fee"], errors="ignore")


shared_cols = sorted(set(green.columns) & set(yellow.columns))


green_aligned = green[shared_cols]
yellow_aligned = yellow[shared_cols]

print(f"\n[RESULT] Shared columns ({len(shared_cols)}) : {shared_cols}")

# %% Concat & Verify Row Count
# CONCEPT: After every merge or concat, verify the row count.
# expected = len(A) + len(B) for a vertical stack.
# An assert here catches silent bugs immediately.


df = pd.concat([green_aligned, yellow_aligned], ignore_index=True)

expected_rows = len(green_aligned) + len(yellow_aligned)
assert (
    df.shape[0] == expected_rows
), f"[ERROR] Row count mismatch: got {df.shape[0]}, expected {expected_rows}"

print(f"[RESULT] Combines: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"[RESULT] Green: {(df['taxi_type'] == 'green').sum():,} rows")
print(f"[RESULT] Yellow: {(df['taxi_type'] == 'yellow').sum():,} rows")
print("[RESULT] Row count assertion passed.")


# %% EDA Checklist
# Always run this before touching the data.
# shape → dtypes → describe → nulls → value_counts


print("=== SHAPE ===")
print(df.shape)

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== DESCRIBE ===")
print(df.describe().T.to_string())

print("\n=== NULL SUMMARY ===")
null_count = df.isnull().sum()
null_pct = (df.isnull().mean() * 100).round(2)
null_summary = pd.DataFrame(
    {
        "null_count": null_count,
        "null_%": null_pct,
    }
).sort_values("null_%", ascending=False)
print(null_summary[null_summary["null_count"] > 0])

print("\n=== NULL % BY TAXI TYPE ===")
null_cols = null_summary[null_summary["null_count"] > 0].index.tolist()
for col in null_cols:
    g_pct = df[df["taxi_type"] == "green"][col].isnull().mean() * 100
    y_pct = df[df["taxi_type"] == "yellow"][col].isnull().mean() * 100
    print(f"  {col:<25}  green={g_pct:.1f}%  yellow={y_pct:.1f}%")


# %% Feature Engineering
# CONCEPT: Raw columns rarely answer questions directly.
# We derive features that are meaningful for analysis.
# Every new feature has a clear formula and a comment explaining why.


# Trip duration in minutes (dropoff - pickup, converted from seconds)

df["trip_duration_min"] = (
    df["dropoff_datetime"] - df["pickup_datetime"]
).dt.total_seconds() / 60

# Hour of day (0–23) and day of week (0=Monday, 6=Sunday)

df["pickup_hour"] = df["pickup_datetime"].dt.hour
df["pickup_dow"] = df["pickup_datetime"].dt.dayofweek

# Speed in mph — guard against zero duration to avoid division by zero

df["speed_mph"] = np.where(
    df["trip_duration_min"] > 0,
    df["trip_distance"] / (df["trip_duration_min"] / 60),
    np.nan,
)

# Tip rate — tip as a fraction of fare (meaningful only for positive fares)

df["tip_rate"] = np.where(
    df["fare_amount"] > 0,
    df["tip_amount"] / df["fare_amount"],
    np.nan,
)

# Map payment_type codes to readable labels

payment_map = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute"}
df["payment_label"] = df["payment_type"].map(payment_map).fillna("Other")

print(
    "[LOG] Features added: trip_duration_min, pickup_hour, pickup_dow, speed_mph, tip_rate, payment_label"
)


# %% Data Cleaning
# CONCEPT: Remove rows that are physically impossible.
# We document every filter and the reason for it.
# Filtering happens AFTER feature engineering so we can use
# the derived columns (trip_duration_min, speed_mph) as filters.


n_before = len(df)

# Duration must be positive and under 5 hours (300 min)

df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 300)]

# Speed above 150 mph is physically impossible in NYC traffic

df = df[(df["speed_mph"].isna()) | (df["speed_mph"] < 150)]

# Fare must be positive (negatives are refunds or data errors)

df = df[df["fare_amount"] > 0]

n_after = len(df)
print(f"[RESULT] Rows removed by cleaning: {n_before - n_after:,}")
print(f"[RESULT] Rows remaining: {n_after:,}")


# %% Q1: What time of day is busiest?


fig, ax = plt.subplots(figsize=(12, 5))

for taxi, color in [("green", "#2ecc71"), ("yellow", "#f1c40f")]:
    hourly = df[df["taxi_type"] == taxi].groupby("pickup_hour").size()
    ax.plot(
        hourly.index,
        hourly.to_numpy(),
        label=taxi.capitalize(),
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4,
    )

ax.set_title("Trip Volume by Hour of Day — Green vs Yellow Taxi", fontsize=13)
ax.set_xlabel("Hour of Day (0 = midnight)")
ax.set_ylabel("Number of Trips")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "q1_trips_by_hour.png", dpi=150)
plt.show()
print("[SAVED] q1_trips_by_hour.png")


# %% Q2: How does median fare compare?


fig, ax = plt.subplots(figsize=(7, 5))

fare_by_type = df.groupby("taxi_type")["fare_amount"].median()
fare_by_type.plot(
    kind="bar",
    ax=ax,
    color=["#2ecc71", "#f1c40f"],
    edgecolor="black",
    width=0.4,
)

ax.set_title("Median Fare Amount — Green vs Yellow Taxi", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Median Fare ($)")
ax.set_xticklabels(["Green", "Yellow"], rotation=0)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "q2_median_fare.png", dpi=150)
plt.show()
print("[SAVED] q2_median_fare.png")


# %% Q3: How does trip distance differ?


fig, ax = plt.subplots(figsize=(10, 5))

for taxi, color in [("green", "#2ecc71"), ("yellow", "#f1c40f")]:
    subset = df[(df["taxi_type"] == taxi) & (df["trip_distance"] < 30)]
    ax.hist(
        subset["trip_distance"],
        bins=60,
        alpha=0.6,
        color=color,
        label=taxi.capitalize(),
        edgecolor="none",
    )

ax.set_title("Trip Distance Distribution (under 30 miles)", fontsize=13)
ax.set_xlabel("Trip Distance (miles)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "q3_trip_distance.png", dpi=150)
plt.show()
print("[SAVED] q3_trip_distance.png")


# %% Q4: Does payment type predict tip rate?


fig, ax = plt.subplots(figsize=(9, 5))

tip_by_payment = (
    df[df["tip_rate"] < 1].groupby("payment_label")["tip_rate"].median() * 100
).sort_values(ascending=False)

tip_by_payment.plot(kind="bar", ax=ax, color="#3498db", edgecolor="black")

ax.set_title("Median Tip Rate (%) by Payment Type", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Median Tip Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "q4_tip_by_payment.png", dpi=150)
plt.show()
print("[SAVED] q4_tip_by_payment.png")


# %% Q5: Does speed vary by hour?


fig, ax = plt.subplots(figsize=(12, 5))

for taxi, color in [("green", "#2ecc71"), ("yellow", "#f1c40f")]:
    subset = df[(df["taxi_type"] == taxi) & (df["speed_mph"] < 80)]
    speed_by_hr = subset.groupby("pickup_hour")["speed_mph"].median()
    ax.plot(
        speed_by_hr.index,
        speed_by_hr.to_numpy(),
        label=taxi.capitalize(),
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4,
    )

ax.set_title("Median Speed (mph) by Hour of Day — NYC Taxis", fontsize=13)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Median Speed (mph)")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "q5_speed_by_hour.png", dpi=150)
plt.show()
print("[SAVED] q5_speed_by_hour.png")


# %% Imputation & Save Clean Dataset
# CONCEPT: Imputation decisions are made AFTER understanding nulls.
# Each decision is documented with its reasoning.
# The clean file is the input for day06_missingness.py and future days.


df_clean = df.copy()

# passenger_count: MAR — vendor-level dropout. Impute with median per taxi_type.

df_clean["passenger_count"] = df_clean.groupby("taxi_type")[
    "passenger_count"
].transform(lambda x: x.fillna(x.median()))

# RatecodeID: overwhelmingly standard rate (1.0). Fill with mode.

df_clean["RatecodeID"] = df_clean["RatecodeID"].fillna(1.0)

# congestion_surcharge: unknown surcharge → assume not applied (0.0)

df_clean["congestion_surcharge"] = df_clean["congestion_surcharge"].fillna(0.0)

# store_and_fwd_flag: unknown → assume not stored/forwarded ('N')

df_clean["store_and_fwd_flag"] = df_clean["store_and_fwd_flag"].fillna("N")

# payment_type: fill with mode (most common payment method)

mode_payment = df_clean["payment_type"].mode()[0]
df_clean["payment_type"] = df_clean["payment_type"].fillna(mode_payment)

# Verify the columns we imputed have no remaining nulls

imputed_cols = [
    "passenger_count",
    "RatecodeID",
    "congestion_surcharge",
    "store_and_fwd_flag",
    "payment_type",
]
remaining_nulls = df_clean[imputed_cols].isnull().sum()
print("[RESULT] Nulls remaining after imputation:")
print(remaining_nulls)

output_path = DATA_PROCESSED + "tlc_combined_clean_jan2023.parquet"
df_clean.to_parquet(output_path, index=False)
print(f"\n[SAVED] {output_path}")
print(f"[RESULT] Final shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")


# %% Summary

print("=" * 55)
print("DAY 06 — STANDARD TASK COMPLETE")
print("=" * 55)
print(f"Green trips: {(df_clean['taxi_type'] == 'green').sum():>10,}")
print(f"Yellow trips: {(df_clean['taxi_type'] == 'yellow').sum():>10,}")
print(f"Total trips: {df_clean.shape[0]:>10,}")
print(f"Columns: {df_clean.shape[1]:>10}")
print(f"Plots saved to: {os.path.abspath(OUTPUTS_DIR)}")
print(f"Clean data saved to: {os.path.abspath(output_path)}")
print("=" * 55)
print("Next: run day06_missingness.py for the out-of-box challenge.")
