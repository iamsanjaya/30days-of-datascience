# main.py — Day 16
# Orchestrates the full Day 16 pipeline:
#   1. Load + split Ames Housing data
#   2. Run 55-trial Optuna study → MLflow logging
#   3. Evaluate best model on validation set
#   4. Run out-of-box challenge: random vs Bayesian (10 trials each)
#   5. Generate all plots
#
# Run: python main.py
# View MLflow dashboard: mlflow ui --backend-store-uri mlruns/

# %%
from pathlib import Path

from sklearn.metrics import mean_squared_error
import numpy as np

from config import OUTPUTS_DIR
from data_loader import load_ames
from optuna_tuner import run_full_study, get_best_model
from random_vs_bayesian import run_comparison
from plots import (
    plot_optimization_history,
    plot_param_importance,
    plot_parallel_coordinates,
    plot_random_vs_bayesian,
)
import warnings

warnings.filterwarnings("ignore")

# %%
# ── 0. Ensure output directory exists ────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ── 1. Load data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Ames Housing data")
print("=" * 60)
X_train, X_val, y_train, y_val = load_ames()

# %%
# ── 2. Run main Optuna study (55 trials, Bayesian / TPE) ─────────────────────
print("\n" + "=" * 60)
print("STEP 2: Running Optuna full study (55 trials)")
print("=" * 60)
study = run_full_study(X_train, y_train)

# %%
# ── 3. Evaluate best model on held-out validation set ────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Evaluating best model on validation set")
print("=" * 60)
best_model = get_best_model(study)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)

val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"  Validation RMSE : ${val_rmse:,.0f}")
print(f"  Best CV RMSE    : ${study.best_value:,.0f}")
print(f"  Best params     : {study.best_trial.params}")

# %%
# ── 4. Out-of-box challenge: Random vs Bayesian (10 trials each) ─────────────
print("\n" + "=" * 60)
print("STEP 4: Out-of-box — Random vs Bayesian (10 trials each)")
print("=" * 60)
random_df, bayesian_df = run_comparison(X_train, y_train)

# %%
# ── 5. Generate all plots ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating plots → outputs/")
print("=" * 60)
plot_optimization_history(study)
plot_param_importance(study)
plot_parallel_coordinates(study)
plot_random_vs_bayesian(random_df, bayesian_df)

print("\n" + "=" * 60)
print("Day 16 complete.")
print("Launch MLflow dashboard: mlflow ui --backend-store-uri mlruns/")
print("=" * 60)
