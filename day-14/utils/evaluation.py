import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from typing import Any


def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    log_t = np.log1p(y_true)
    log_p = np.log1p(y_pred)

    return float(np.sqrt(mean_squared_error(log_t, log_p)))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_rmsle(model: Any, X: pd.DataFrame, y: pd.Series, n_folds=5, random_state=42):
    log_y = np.log1p(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores = cross_val_score(
        model,
        X,
        log_y,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    scores = -scores

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
    }


def print_cv_result(label, result):
    print(f"\n{label}")
    print(f"RMSLE: {result['mean']:.5f} (+/- {result['std']:.5f})")
    print("Folds:", " ".join(f"{x:.5f}" for x in result["scores"]))


def improvement_pct(base, new):
    return 100 * (base - new) / base
