"""
04_label_corruption_test.py — Day 17, Stress Test 3/3

Randomly selects 10% of test rows and shuffles their SalePrice values among
each other (a permutation within the subset, not synthetic values — every
number used is a real SalePrice that existed somewhere in the test set,
just attached to the wrong row). This tests how much the model's apparent
performance depends on having clean ground truth, independent of the
feature-noise question covered in 03.
"""

import json

import joblib
import numpy as np

import config


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmsle(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true)) ** 2)))


def corrupt_labels(y, fraction: float, random_state: int):
    rng = np.random.RandomState(random_state)
    y_corrupted = y.to_numpy().copy()
    n = len(y_corrupted)
    n_corrupt = int(round(fraction * n))
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    shuffled_values = y_corrupted[corrupt_idx].copy()
    rng.shuffle(shuffled_values)
    y_corrupted[corrupt_idx] = shuffled_values
    return y_corrupted, corrupt_idx


def main():
    pipeline = joblib.load(config.BEST_PIPELINE_PATH)
    X_test, y_test = joblib.load(config.TEST_SPLIT_PATH)

    y_true = y_test.to_numpy()
    y_corrupted, corrupt_idx = corrupt_labels(
        y_test, config.LABEL_CORRUPTION_FRACTION, config.RANDOM_STATE
    )

    # Note: predictions don't change (features untouched) — what changes is
    # the ground truth we're scoring against. This measures how much the
    # *reported* metric would mislead you if 10% of labels were wrong,
    # not a retrained model's behavior.
    y_pred = pipeline.predict(X_test)

    results = {
        "corruption_fraction": config.LABEL_CORRUPTION_FRACTION,
        "n_labels_corrupted": int(len(corrupt_idx)),
        "clean_labels": {
            "rmse": rmse(y_true, y_pred),
            "rmsle": rmsle(y_true, y_pred),
        },
        "corrupted_labels": {
            "rmse": rmse(y_corrupted, y_pred),
            "rmsle": rmsle(y_corrupted, y_pred),
        },
    }
    results["rmse_degradation_pct"] = (
        (results["corrupted_labels"]["rmse"] - results["clean_labels"]["rmse"])
        / results["clean_labels"]["rmse"]
        * 100
    )

    config.CORRUPTION_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
