"""
03_noise_injection_test.py — Day 17, Stress Test 2/3

Adds Gaussian noise to numeric test features only, scaled to each column's
own standard deviation (so a 20% noise level means roughly +/-20% of that
column's natural spread, not a 20% chance of corruption). Categorical
features and the target are left untouched — this test isolates sensitivity
to measurement-quality degradation in the inputs, not label quality
(that's 04_label_corruption_test.py's job).
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


def add_noise(X, noise_level: float, random_state: int):
    """Return a copy of X with Gaussian noise added to numeric columns only."""
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_std = X_noisy[col].std()
        if col_std == 0 or np.isnan(col_std):
            continue
        noise = rng.normal(loc=0.0, scale=noise_level * col_std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    return X_noisy, list(numeric_cols)


def main():
    pipeline = joblib.load(config.BEST_PIPELINE_PATH)
    X_test, y_test = joblib.load(config.TEST_SPLIT_PATH)

    X_noisy, numeric_cols = add_noise(
        X_test, config.FEATURE_NOISE_LEVEL, config.RANDOM_STATE
    )

    y_pred_clean = pipeline.predict(X_test)
    y_pred_noisy = pipeline.predict(X_noisy)
    y_true = y_test.to_numpy()

    results = {
        "noise_level": config.FEATURE_NOISE_LEVEL,
        "n_numeric_cols_perturbed": len(numeric_cols),
        "clean": {
            "rmse": rmse(y_true, y_pred_clean),
            "rmsle": rmsle(y_true, y_pred_clean),
        },
        "noisy": {
            "rmse": rmse(y_true, y_pred_noisy),
            "rmsle": rmsle(y_true, y_pred_noisy),
        },
    }
    results["rmse_degradation_pct"] = (
        (results["noisy"]["rmse"] - results["clean"]["rmse"])
        / results["clean"]["rmse"]
        * 100
    )

    config.NOISE_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
