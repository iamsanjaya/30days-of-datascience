"""
06_visualize_results.py — Day 17

Two plots, chosen because they're the two things this day's results
actually need to show, not decoration:

1. robustness_comparison.png — bar chart of RMSE across baseline, the two
   distribution-shift slices, feature noise, and label corruption. This is
   the single chart that makes "label corruption hurt ~4x more than feature
   noise" visible at a glance.

2. predicted_vs_actual_by_era.png — scatter of predicted vs. actual sale
   price on the test set, colored by old/new home (the same 1960 split used
   in 02). Makes the RMSE-vs-RMSLE disagreement concrete: visually shows
   whether errors cluster differently across the price range for each slice.

Reads everything from outputs/*.json and models/*.joblib — does not refit
or reload data, so it's safe to re-run after any of 01-05.
"""

import json

import joblib
import matplotlib.pyplot as plt

import config

PLOTS_DIR = config.OUTPUT_DIR / "plots"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_robustness_comparison():
    baseline = load_json(config.BASELINE_METRICS_PATH)
    shift = load_json(config.DISTRIBUTION_SHIFT_RESULTS_PATH)
    noise = load_json(config.NOISE_RESULTS_PATH)
    corruption = load_json(config.CORRUPTION_RESULTS_PATH)

    labels = [
        "Baseline\n(clean)",
        f"Old homes\n(< {shift['threshold_year']})",
        f"New homes\n(>= {shift['threshold_year']})",
        f"Feature noise\n({noise['noise_level']*100:.0f}%)",
        f"Label corruption\n({corruption['corruption_fraction']*100:.0f}%)",
    ]
    rmse_values = [
        baseline["rmse"],
        shift["old_homes"]["rmse"],
        shift["new_homes"]["rmse"],
        noise["noisy"]["rmse"],
        corruption["corrupted_labels"]["rmse"],
    ]
    colors = ["#4C72B0", "#DD8452", "#DD8452", "#C44E52", "#8C0000"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, rmse_values, color=colors)
    ax.axhline(
        baseline["rmse"], color="#4C72B0", linestyle="--", linewidth=1, alpha=0.6
    )
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Label corruption degraded the model ~4x more than feature noise")
    for bar, val in zip(bars, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(rmse_values) * 0.015,
            f"${val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path = PLOTS_DIR / "robustness_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_predicted_vs_actual_by_era():
    pipeline = joblib.load(config.BEST_PIPELINE_PATH)
    X_test, y_test = joblib.load(config.TEST_SPLIT_PATH)

    y_pred = pipeline.predict(X_test)
    y_true = y_test.to_numpy()
    old_mask = (X_test[config.SHIFT_COLUMN] < config.SHIFT_YEAR_THRESHOLD).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        y_true[old_mask],
        y_pred[old_mask],
        alpha=0.6,
        s=20,
        color="#DD8452",
        label=f"Old homes (< {config.SHIFT_YEAR_THRESHOLD}), n={old_mask.sum()}",
    )
    ax.scatter(
        y_true[~old_mask],
        y_pred[~old_mask],
        alpha=0.6,
        s=20,
        color="#4C72B0",
        label=f"New homes (>= {config.SHIFT_YEAR_THRESHOLD}), n={(~old_mask).sum()}",
    )
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(
        lims,
        lims,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Perfect prediction",
    )
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title("Predicted vs. Actual by build era — test set")
    ax.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / "predicted_vs_actual_by_era.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_robustness_comparison()
    plot_predicted_vs_actual_by_era()


if __name__ == "__main__":
    main()
