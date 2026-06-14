# %%
"""
Chart EDA visual report on NYC TLC Green Taxi data.
Each chart answers one specific question.
Each title is a conclusion, not a description.
Charts saved to: day-07/outputs/
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.ticker as mticker
import seaborn as sns

import config  # applies theme + exposes paths and titles

# Data loading

# %%
df = pd.read_parquet(config.DATA_PATH)

# Ensure datetime column is parsed

if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# pickup_hour and pickup_dow already exist from Day 06 feature engineering
DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")


# Chart helpers


def save_chart(fig: matplotlib.figure.Figure, filename: str) -> None:
    """Save figure to OUTPUT_DIR and close it."""
    path = config.OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# Chart 1 — Hourly demand
# Question: Where is trip demand concentrated through the day?
# Conclusion: Demand peaks at 8 AM and 6 PM - midday and night are quiet.

# %%


def chart_hourly_demand(df: pd.DataFrame) -> None:
    hourly = df.groupby("pickup_hour").size().reset_index(name="trip_count")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        hourly["pickup_hour"],
        hourly["trip_count"],
        color="steelblue",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        hourly["pickup_hour"], hourly["trip_count"], alpha=0.15, color="steelblue"
    )
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day (0 = midnight)")
    ax.set_ylabel("Number of Trips")
    ax.set_title(config.TITLE_HOURLY_DEMAND)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0)  # always start Y at zero for honest baseline

    fig.tight_layout()
    save_chart(fig, "01_hourly_demand.png")


# Chart 2 — Fare vs trip distance scatter
# Question: Do longer trips earn proportionally more per mile?
# Conclusion: Fare grows with distance, but short trips earn more per mile.


# %%
def chart_fare_distance(df: pd.DataFrame) -> None:
    # Sample for readability: full dataset overplots into a blob
    sample = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 30)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] < 80)
    ].sample(n=min(8_000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        sample["trip_distance"],
        sample["fare_amount"],
        alpha=0.15,
        s=6,
        color="steelblue",
        rasterized=True,
    )

    # OLS trend line via numpy
    coef = np.polyfit(sample["trip_distance"], sample["fare_amount"], deg=1)
    x_range = np.linspace(0, 30, 100)
    ax.plot(
        x_range,
        np.polyval(coef, x_range),
        color="crimson",
        linewidth=1.8,
        label="OLS trend",
    )
    ax.legend(fontsize=9)
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_title(config.TITLE_FARE_DISTANCE)

    fig.tight_layout()
    save_chart(fig, "02_fare_vs_distance.png")


# Chart 3 — Average tip amount by payment type
# Question: Do card passengers tip more than cash passengers?
# Conclusion: Card payments generate 5x more recorded tip revenue than cash.
# Note: Cash tips are recorded as 0.0 in TLC data, not as missing -
#       this is MNAR (Missing Not At Random), not genuine zero-tip behaviour.


# %%
def chart_payment_tip(df: pd.DataFrame) -> None:
    payment_map = {
        1: "Credit Card",
        2: "Cash",
        3: "No Charge",
        4: "Dispute",
        5: "Unknown",
    }
    df_pay = df.copy()
    df_pay["payment_label"] = df_pay["payment_type"].map(payment_map).fillna("Other")

    tip_by_payment = (
        df_pay[df_pay["payment_label"].isin(["Credit Card", "Cash"])]
        .groupby("payment_label")["tip_amount"]
        .mean()
        .reset_index(name="avg_tip")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=tip_by_payment,
        x="payment_label",
        y="avg_tip",
        hue="payment_label",
        palette={"Credit Card": "steelblue", "Cash": "coral"},
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Payment Type")
    ax.set_ylabel("Average Tip Amount ($)")
    ax.set_title(config.TITLE_PAYMENT_TIP)
    ax.set_ylim(0)
    ax.annotate(
        "Cash tip = $0.00 in TLC records — MNAR, not genuine zero-tip behavior",
        xy=(0.5, 0.96),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8,
        color="grey",
    )

    fig.tight_layout()
    save_chart(fig, "03_payment_tip.png")


# Chart 4 — Tip amount by day of week
# Question: Are tips a weekend phenomenon or consistent across the week?
# Conclusion: Weekend tips are higher and more variable than weekday tips.


# %%
def chart_tip_by_dow(df: pd.DataFrame) -> None:
    # Only card payments generate recorded tip data in TLC
    df_tips = df[(df["payment_type"] == 1) & (df["tip_amount"] >= 0)].copy()

    fig, ax = plt.subplots(figsize=(9, 4))

    # FIX: Map the integers 0-6 to steelblue since pickup_dow is numeric
    dow_palette = {i: "steelblue" for i in range(7)}

    sns.boxplot(
        data=df_tips,
        x="pickup_dow",
        y="tip_amount",
        hue="pickup_dow",
        palette=dow_palette,
        dodge=False,  # Prevents Seaborn from narrowing/shifting the boxes
        legend=False,  # Prevents a redundant day-of-week legend from appearing
        ax=ax,
        flierprops=dict(marker=".", markersize=2, alpha=0.2),
        medianprops=dict(color="crimson", linewidth=2),
    )

    # Map the x-axis tick positions (0-6) to the actual day names
    ax.set_xticks(range(7))
    ax.set_xticklabels(DOW_ORDER)

    # Matplotlib's patch_artist equivalent in Seaborn to handle box opacity
    for patch in ax.patches:
        patch.set_alpha(0.6)

    ax.set_ylim(0, 20)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Tip Amount ($)")
    ax.set_title(config.TITLE_TIP_BY_DOW)
    ax.annotate(
        "Capped at $20 for readability — outliers beyond this exist",
        xy=(0.01, 0.96),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
        va="top",
    )

    fig.tight_layout()
    save_chart(fig, "04_tip_by_dow.png")


# Chart 5 — Vendor trip volume: peak vs off-peak
# Question: How does vendor trip volume compare in peak vs off-peak hours?
# Conclusion: Vendor 2 handles more peak-hour volume than Vendor 1.


# %%
def chart_vendor_peak(df: pd.DataFrame) -> None:
    PEAK_HOURS = {7, 8, 9, 17, 18, 19}  # morning + evening rush
    df_vendor = df.copy()
    df_vendor["period"] = df_vendor["pickup_hour"].apply(
        lambda h: "Peak" if h in PEAK_HOURS else "Off-Peak"
    )
    df_vendor["vendor_label"] = df_vendor["VendorID"].map(
        {1: "Vendor 1", 2: "Vendor 2"}
    )

    counts = (
        df_vendor.groupby(["vendor_label", "period"])
        .size()
        .reset_index(name="trip_count")
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=counts,
        x="vendor_label",
        y="trip_count",
        hue="period",
        palette={"Peak": "coral", "Off-Peak": "steelblue"},
        ax=ax,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Vendor")
    ax.set_ylabel("Number of Trips")
    ax.set_title(config.TITLE_VENDOR_PEAK)
    ax.legend(title="Time Period")

    fig.tight_layout()
    save_chart(fig, "05_vendor_peak.png")


# Chart 6 — Fare amount distribution
# Question: What does the full fare distribution look like?
# Conclusion: Most fares cluster under $20 — high-fare outliers are rare.


# %%
def chart_fare_distribution(df: pd.DataFrame) -> None:
    df_fare = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 100)]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=df_fare,
        x="fare_amount",
        bins=60,
        kde=True,
        color="steelblue",
        line_kws={"linewidth": 1.6},
        ax=ax,
    )
    median_fare = df_fare["fare_amount"].median()
    ax.axvline(median_fare, color="crimson", linewidth=1.4, linestyle="--")
    ax.text(
        median_fare + 0.5,
        ax.get_ylim()[1] * 0.85,
        f"Median: ${median_fare:.2f}",
        color="crimson",
        fontsize=9,
    )
    ax.set_xlabel("Fare Amount ($)")
    ax.set_ylabel("Number of Trips")
    ax.set_title(config.TITLE_FARE_DIST)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    save_chart(fig, "06_fare_distribution.png")


# Run all charts


# %%
if __name__ == "__main__":
    print("Generating 6-chart EDA report...")
    chart_hourly_demand(df)
    chart_fare_distance(df)
    chart_payment_tip(df)
    chart_tip_by_dow(df)
    chart_vendor_peak(df)
    chart_fare_distribution(df)
    print("\nAll charts saved to:", config.OUTPUT_DIR)
