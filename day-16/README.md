# Day 16 — Hyperparameter Tuning with Optuna + MLflow

## What This Day Covers

Standard task: tune XGBoost on the Ames Housing dataset using Optuna (55 trials, Bayesian / TPE sampler), log every trial to MLflow, and visualize the optimization process three ways.

Out-of-box challenge: empirically test whether Bayesian optimization actually beats random search (50 trials each from the same seed), and label each Bayesian trial as exploration or exploitation.

---

## File Structure

```
day-16/
├── config.py              # All constants, paths, search space (single source of truth)
├── data_loader.py         # Load Ames Housing, clean, one-hot encode, split
├── optuna_tuner.py        # 55-trial Optuna study with MLflow child run logging
├── random_vs_bayesian.py  # Out-of-box: random vs Bayesian trials, exploration labels
├── plots.py               # All visualization logic (no plot calls elsewhere)
├── main.py                # Orchestrator: runs all steps in order
├── README.md
├── data/                  # gitignored
|   └── train.csv          # Ames Housing (from Kaggle)
└── outputs/               # gitignored
    ├── optimization_history.png
    ├── param_importance.png
    ├── parallel_coordinates.png
    └── random_vs_bayesian.png

learning-journal/
└── day-16.md

mlruns/         ← gitignored — MLflow tracking store
```

---

## Setup

```bash
pip install optuna mlflow xgboost scikit-learn pandas numpy matplotlib seaborn
```

Place `train.csv` (Ames Housing) inside `data/`.

---

## How to Run

```bash
cd ~/Desktop/projects/day-16
python main.py
```

Then launch the MLflow dashboard to inspect all logged runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` in your browser.

---

## Standard Task Results

| Metric                    | Value                                                           |
| ------------------------- | --------------------------------------------------------------- |
| Dataset                   | Ames Housing — 2,930 rows, 254 features after encoding          |
| Train / Val split         | 2,344 / 586 rows (80/20)                                        |
| Dropped high-missing cols | Alley, Mas Vnr Type, Fireplace Qu, Pool QC, Fence, Misc Feature |
| Optuna trials             | 55 (TPE / Bayesian sampler)                                     |
| CV folds                  | 5-fold, scoring = RMSE                                          |
| Best CV RMSE              | $22,045                                                         |
| Validation RMSE           | $24,772                                                         |

The ~$2,700 gap between CV RMSE and validation RMSE is expected — CV averages over 5 folds on training data, while validation is a true holdout. The gap is ~12%, which does not indicate severe overfitting.

**Best hyperparameters:**

```
n_estimators      : 829
max_depth         : 3
learning_rate     : 0.0371
subsample         : 0.559
colsample_bytree  : 0.859
min_child_weight  : 6
reg_alpha         : 6.24e-05
reg_lambda        : 0.00369
gamma             : 0.862
```

**Notable pattern:** The best model uses shallow trees (`max_depth=3`) with many estimators (`n_estimators=829`) and a low learning rate (`0.037`). This is the classic XGBoost configuration — many weak learners each correcting a small residual, rather than a few deep trees attempting to capture everything at once.

---

## Out-of-Box Challenge: Random vs Bayesian

**Question:** Does Bayesian optimization (TPE) actually outperform random search, and by how much?

**Methodology:**

- Both samplers initialized with the same `RANDOM_STATE` seed so starting conditions are identical.
- Random sampler: uniform sampling across the search space with no memory of past trials.
- Bayesian sampler (TPE): builds a probabilistic surrogate model and biases subsequent trials toward promising regions.
- Exploration vs exploitation labels: a trial is "exploitation" if it improves on the running best RMSE; otherwise "exploration."

**Results:**

| Sampler        | Best RMSE              |
| -------------- | ---------------------- |
| Random Search  | $22,299                |
| Bayesian (TPE) | $22,045                |
| Winner         | Bayesian (TPE) by $254 |

**Key finding:**

Bayesian won, but only by $254. The margin is narrow enough that a different random seed could reverse the result. This is the honest answer to the challenge question — Bayesian optimization is not automatically superior. Its advantage compounds as the trial count grows, because the surrogate model needs sufficient data to reliably distinguish good parameter regions from bad ones. At lower trial counts, random search remains a legitimate and computationally cheaper competitor.

---

## Plots Produced

| File                       | What it shows                                                                 |
| -------------------------- | ----------------------------------------------------------------------------- |
| `optimization_history.png` | Per-trial RMSE scatter + running best curve across 55 trials                  |
| `param_importance.png`     | fANOVA importance scores — which hyperparameters drove RMSE change most       |
| `parallel_coordinates.png` | Multi-axis view of top 6 params, colored by RMSE (green = good trials)        |
| `random_vs_bayesian.png`   | Running best comparison + exploration/exploitation bar chart for Bayesian arm |

---

## MLflow Runs

Each of the 55 trials is logged as a **nested child run** under a parent `full-study-parent` run in the `day16-xgboost-optuna-ames` experiment. The random and Bayesian comparison runs are logged as separate top-level runs.

Inside each trial run: all hyperparameter values, CV RMSE, and CV RMSE standard deviation.

---

## Key Concepts Reinforced

- **TPE (Tree-structured Parzen Estimator):** models the distribution of good and bad parameter regions separately, then samples from the ratio. Requires a warm-up period before the surrogate becomes reliable.
- **fANOVA parameter importance:** measures how much variance in objective value is explained by each hyperparameter across the full search space. More reliable than ranking by best trial alone.
- **Exploration vs exploitation:** every optimization algorithm balances trying new regions (exploration) vs refining known good regions (exploitation). The empirical labeling in the comparison study makes this tradeoff visible.
- **MLflow nested runs:** child runs allow grouping all trials under one parent for clean experiment comparison in the dashboard.
