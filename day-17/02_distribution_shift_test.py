"""
02_distribution_shift_test.py — Day 17, Stress Test 1/3

Distribution shift, defined precisely: split the already-held-out test set
(never touched during training either way) into two slices by build era —
"old" (Year Built < SHIFT_YEAR_THRESHOLD) and "new" (>= threshold) — and
compare the frozen pipeline's performance on each slice.

This stays leakage-free since both slices come from data the model never
trained on. It tests whether the model generalizes evenly across the
population, or whether it's quietly tuned to one era.
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


def evaluate_slice(pipeline, X_slice, y_slice):
    if len(y_slice) == 0:
        return None
    y_pred = pipeline.predict(X_slice)
    return {
        "n": int(len(y_slice)),
        "rmse": rmse(y_slice.values, y_pred),
        "rmsle": rmsle(y_slice.values, y_pred),
    }


def main():
    pipeline = joblib.load(config.BEST_PIPELINE_PATH)
    X_test, y_test = joblib.load(config.TEST_SPLIT_PATH)

    old_mask = X_test[config.SHIFT_COLUMN] < config.SHIFT_YEAR_THRESHOLD
    new_mask = ~old_mask

    results = {
        "shift_column": config.SHIFT_COLUMN,
        "threshold_year": config.SHIFT_YEAR_THRESHOLD,
        "overall": evaluate_slice(pipeline, X_test, y_test),
        "old_homes": evaluate_slice(pipeline, X_test[old_mask], y_test[old_mask]),
        "new_homes": evaluate_slice(pipeline, X_test[new_mask], y_test[new_mask]),
    }

    config.DISTRIBUTION_SHIFT_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
