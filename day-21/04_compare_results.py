"""
Day 21 — Step 4: Compare Frozen vs Fine-Tuned on the Held-Out Test Set

Evaluates both saved models on the test split (never seen during
training or validation) and reports accuracy, precision, recall, F1,
and AUC side by side — the documented "accuracy jump" deliverable.

Run:
    python 04_compare_results.py
"""

import json


import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from utils import data, visualization


def evaluate(model: tf.keras.Model, test_ds: tf.data.Dataset, test_df) -> dict:
    y_true = test_df["label"].values
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
    }


def main() -> None:
    print(f"Backbone: {config.BACKBONE}")
    _, _, test_df = data.load_splits()
    test_ds = data.build_dataset(test_df, augment=False, shuffle=False)

    print(f"Loading frozen model from {config.FROZEN_MODEL_PATH}...")
    frozen_model = tf.keras.models.load_model(config.FROZEN_MODEL_PATH)
    print(f"Loading fine-tuned model from {config.FINETUNED_MODEL_PATH}...")
    finetuned_model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)

    print("Evaluating frozen model on test set...")
    frozen_metrics = evaluate(frozen_model, test_ds, test_df)
    print("Evaluating fine-tuned model on test set...")
    finetuned_metrics = evaluate(finetuned_model, test_ds, test_df)

    results = {"frozen": frozen_metrics, "finetuned": finetuned_metrics}

    config.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.COMPARISON_DIR / "test_set_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Test Set Comparison ---")
    print(f"{'Metric':<12}{'Frozen':>10}{'Fine-Tuned':>12}{'Delta':>10}")
    for metric_name in frozen_metrics:
        frozen_val = frozen_metrics[metric_name]
        finetuned_val = finetuned_metrics[metric_name]
        delta = finetuned_val - frozen_val
        print(
            f"{metric_name:<12}{frozen_val:>10.4f}{finetuned_val:>12.4f}{delta:>+10.4f}"
        )

    visualization.plot_metric_comparison(
        results,
        save_path=config.COMPARISON_DIR / "frozen_vs_finetuned_metrics.png",
    )
    print(f"\nResults saved to {results_path}")
    print(f"Comparison plot saved to {config.COMPARISON_DIR}")


if __name__ == "__main__":
    main()
