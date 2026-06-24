"""
Day 21 — Step 6 (Xception extension): Compare Backbones

Run the full pipeline (02 -> 05) once with config.BACKBONE = "resnet50"
and once with config.BACKBONE = "xception" (change the setting in
config.py between runs, since each run reads it once at import time).
This script then loads both saved test-set metric files and reports
them side by side — accuracy, precision, recall, F1, AUC.

Run (after both backbones have completed 04_compare_results.py):
    python 06_compare_backbones.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import config


def main() -> None:
    results = {}
    for backbone in ("resnet50", "xception"):
        metrics_path = (
            config.OUTPUT_DIR / "comparison" / backbone / "test_set_metrics.json"
        )
        if not metrics_path.exists():
            print(f"Missing: {metrics_path}")
            print(
                f'Run the full pipeline with config.BACKBONE = "{backbone}" first '
                f"(02_train_frozen.py -> 04_compare_results.py)."
            )
            continue
        with open(metrics_path, "r") as f:
            results[backbone] = json.load(f)["finetuned"]

    if len(results) < 2:
        print("\nNeed both backbones' results before comparing. Stopping.")
        return

    print("\n--- Backbone Comparison (fine-tuned, test set) ---")
    metric_names = list(next(iter(results.values())).keys())
    print(f"{'Metric':<12}{'ResNet50':>12}{'Xception':>12}{'Delta':>10}")
    for metric_name in metric_names:
        resnet_val = results["resnet50"][metric_name]
        xception_val = results["xception"][metric_name]
        delta = xception_val - resnet_val
        print(
            f"{metric_name:<12}{resnet_val:>12.4f}{xception_val:>12.4f}{delta:>+10.4f}"
        )

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, backbone in enumerate(("resnet50", "xception")):
        values = [results[backbone][m] for m in metric_names]
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=backbone)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("ResNet50 vs Xception — Fine-Tuned Test Set Metrics")
    ax.legend()
    fig.tight_layout()

    save_path = config.OUTPUT_DIR / "comparison" / "backbone_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"\nComparison plot saved to {save_path}")


if __name__ == "__main__":
    main()
