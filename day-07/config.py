# %%
"""
Single source of truth for Day 07 - all paths, chart titles, and plot style.

Importing this module applies the Seaborn theme globally.
No script should hardcode paths or title strings.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Paths

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = (
    BASE_DIR.parent
    / "day-06"
    / "data"
    / "processed"
    / "tlc_combined_clean_jan2023.parquet"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Chart titles - every title is a conclusion, nto a label

TITLE_HOURLY_DEMAND = "Demand peaks at 8 AM and 6 PM - midday and night are quiet."
TITLE_FARE_DISTANCE = "Fare grows with distance, but short trips earn more per mile."
TITLE_PAYMENT_TIP = "Card payments generate 5x more recorded tip revenue than cash"
TITLE_TIP_BY_DOW = "Weekend tips are higher and more variable than weekday tips."
TITLE_VENDOR_PEAK = "Vendor 2 handles more peak-hour volume than vendor 1."
TITLE_FARE_DIST = "Most fares cluster under $20 - high-fare outliers are rare."
TITLE_MISLEADING = "Misleading: trip demand 'Surges' 38% at rush hour."
TITLE_HONEST = "Honest: rush-hour demand is only modestly higher than baseline."


# Plot style - applied once on import

PALETTE = "muted"
FIGURE_DPI = 300
FONT_SCALE = 1.1

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=FONT_SCALE)
plt.rcParams["figure.dpi"] = 100  # screen preview
plt.rcParams["savefig.dpi"] = FIGURE_DPI
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
