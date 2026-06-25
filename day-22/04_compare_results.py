# %%
"""
04_compare_results.py — Standard Task

Generate the two output plots for the standard task:
  1. Data efficiency degradation curve (accuracy vs % training data)
  2. Augmentation effect comparison on the smallest subsets
"""

import json

import config
from utils.visualization import plot_augmentation_comparison, plot_data_efficiency_curve

# %%
with open(config.DATA_EFFICIENCY_RESULTS_PATH) as f:
    efficiency_results = json.load(f)

plot_data_efficiency_curve(
    efficiency_results,
    config.PLOTS_DIR / "data_efficiency_curve.png",
)
print("Saved data_efficiency_curve.png")

# %%
try:
    with open(config.AUGMENTATION_RESULTS_PATH) as f:
        augmentation_results = json.load(f)

    plot_augmentation_comparison(
        augmentation_results,
        config.PLOTS_DIR / "augmentation_comparison.png",
    )
    print("Saved augmentation_comparison.png")
except FileNotFoundError:
    print(
        "No augmentation results found yet — run 03_augmentation_experiment.py first."
    )

print("\nAll standard-task plots saved to", config.PLOTS_DIR)
