# %%
"""
11_compare_niche_results.py — Out-of-Box Challenge: final comparison

Aggregate every technique's result from niche_results.json into a comparison
table (saved as CSV) and bar chart, so the strategy document can cite exact
numbers.
"""

import json

import pandas as pd

import config
from utils.visualization import plot_niche_comparison

# %%
with open(config.NICHE_RESULTS_PATH) as f:
    results = json.load(f)

summary_rows = [
    {"technique": technique, "test_accuracy": metrics["accuracy"]}
    for technique, metrics in results.items()
]
summary_df = pd.DataFrame(summary_rows).sort_values("test_accuracy", ascending=False)
print(summary_df.to_string(index=False))

summary_path = config.RESULTS_DIR / "niche_summary_table.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary table to {summary_path}")

# %%
plot_niche_comparison(results, config.PLOTS_DIR / "niche_techniques_comparison.png")
print("Saved niche_techniques_comparison.png")

best = summary_df.iloc[0]
print(
    f"\nBest technique: {best['technique']} with {best['test_accuracy']:.2%} test accuracy "
    f"on a 450-image training pool."
)
