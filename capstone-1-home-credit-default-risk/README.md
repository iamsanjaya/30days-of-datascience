# Capstone 1 — Home Credit Default Risk

End-to-end structured-data ML project predicting loan default risk
(`TARGET`) using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
Kaggle competition dataset. Built as Capstone 1 of a 30-day Data
Science / ML internship-prep roadmap (Days 24–25: Structured Data).

## Results Summary

| Metric                                  | Value            |
| --------------------------------------- | ---------------- |
| **Held-out test ROC-AUC**               | **0.7712**       |
| 95% bootstrap CI (1000 resamples)       | [0.7650, 0.7777] |
| Tuned model — 5-fold CV ROC-AUC         | 0.7666 ± 0.0012  |
| Baseline (untuned) LightGBM CV ROC-AUC  | 0.7617           |
| Baseline Logistic Regression CV ROC-AUC | 0.7489           |
| Baseline Random Forest CV ROC-AUC       | 0.7223           |

**Top 5 features by SHAP importance:** `EXT_SOURCE_3`, `EXT_SOURCE_2`,
`CREDIT_TERM_RATIO`, `EXT_SOURCE_1`, `GOODS_PRICE_CREDIT_RATIO`.

See [`risk_report.md`](./risk_report.md) for a critical assessment of
where and why this model could fail in production — not just why it
works.

## Pipeline

Scripts run in order; each reads the previous step's saved output rather
than recomputing it, so any single step can be re-run independently once
its input exists on disk.

| Script                         | Purpose                                                                                                                                      | Output                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `01_data_prep.py`              | Load raw `application_train.csv`, validate shape/duplicates, clean the `DAYS_EMPLOYED` sentinel anomaly (365243 → null + flag)               | `application_train_clean.parquet`                                               |
| `02_eda.py`                    | 4 targeted findings (not exhaustive EDA) on the cleaned data                                                                                 | 4 PNGs in `outputs/eda/`                                                        |
| `03_feature_engineering.py`    | Row-wise engineered features (ratios, age/tenure conversions, EXT_SOURCE missingness flags) — no fitted statistics, so safe to run pre-split | `application_train_features.parquet`                                            |
| `04_train_baseline_models.py`  | Compare Logistic Regression, Random Forest, LightGBM via identical sklearn Pipelines, Stratified 5-fold CV                                   | `model_comparison.csv`                                                          |
| `05_tune_best_model_optuna.py` | Optuna-tune the winning model (LightGBM) — 50 trials, 3-fold CV search, re-evaluated at full 5-fold CV; MLflow logging (SQLite backend)      | `lightgbm_tuned.joblib`, `mlflow.db`                                            |
| `06_final_eval_shap.py`        | Load the tuned model (no retraining), bootstrap CI on test ROC-AUC, SHAP TreeExplainer top-5 feature analysis                                | `bootstrap_ci_results.csv`, `shap_top5_features.csv`, 2 PNGs in `outputs/shap/` |

### 01 — Data Prep

Loads the raw CSV, validates it against the known official shape
(307,511 rows × 122 columns), checks for duplicate `SK_ID_CURR` values
(expected: zero — one row per loan application), and handles
`DAYS_EMPLOYED`'s sentinel value of `365243`, which Home Credit uses to
mean "not applicable" (pensioners/unemployed applicants) rather than a
literal ~1,000 years of employment. This gets nulled out and flagged
(`DAYS_EMPLOYED_ANOM`) rather than left as a numeric outlier that would
silently distort any model trained on the raw value.

### 02 — EDA

Four targeted questions, chosen specifically to go beyond the standard
public-notebook treatment of this dataset:

1. **Does the `DAYS_EMPLOYED` anomaly flag correlate with default rate?**
   Confirmed: anomalous applicants default _less_ often than the overall
   population (5.4% vs. an 8.07% overall baseline — the non-anomalous
   group specifically defaults at 8.66%, higher than the overall rate
   because the safer pensioner group pulls the aggregate down) — counter
   to a naive "no current employment = higher risk" assumption.
2. **Does missingness in `EXT_SOURCE_1/2/3` itself carry signal**,
   independent of the score values? _(Methodology confirmed in code;
   exact percentages pending — see risk report.)_
3. **Is the credit-to-income ratio's relationship with default linear,
   or noisier than commonly assumed?** Examined via decile binning
   rather than fixed cutoffs, letting the data define "low/medium/high."
   _(Which decile peaked — pending exact run output.)_
4. **Is the `DAYS_EMPLOYED` anomaly really a pension-income-stability
   effect**, or a different underlying pattern? Cross-tabulated against
   `NAME_INCOME_TYPE`, with a minimum group-size filter (n ≥ 50) to avoid
   treating tiny categories' default rates as meaningful. _(Exact
   cross-tab results pending — see risk report, Section 3.)_

### 03 — Feature Engineering

Adds purely row-wise features — no fitted statistics, so this step is
safe to run before the train/test split without leakage:

- `AGE_YEARS`, `YEARS_EMPLOYED`, `YEARS_REGISTRATION`, `YEARS_ID_PUBLISH`
  — sign-flipped, year-converted versions of the raw `DAYS_*` columns.
- `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM_RATIO`,
  `GOODS_PRICE_CREDIT_RATIO` — financial ratios. Division-by-zero cases
  are caught explicitly and converted from `inf` to `NaN` so they get
  handled by the pipeline's imputer downstream, rather than silently
  corrupting the model with infinite values.
- `EXT_SOURCE_1_MISSING`, `EXT_SOURCE_2_MISSING`, `EXT_SOURCE_3_MISSING`
  — boolean flags marking missingness in the three external bureau
  score columns. The underlying `EXT_SOURCE_*` values are left as `NaN`
  here deliberately; imputing them requires a fitted statistic (e.g.
  median), which belongs inside the Pipeline (fit only on training
  folds) to avoid leakage.

Imputation, categorical encoding, and scaling are all deliberately
**deferred** to the Pipeline objects in `04` and `05` — fitting any of
those before the train/test split would risk leaking test-set
information into preprocessing.

### 04 — Baseline Model Comparison

Three algorithms, each wrapped in an identical-preprocessing sklearn
Pipeline (median imputation for numeric columns; most-frequent imputation

- one-hot encoding for categorical columns, with unknown categories
  ignored rather than erroring):

* **Logistic Regression** — adds `StandardScaler(with_mean=False)` (no
  centering, since the one-hot output is sparse and centering would
  force a dense conversion at this column count).
* **Random Forest** (200 estimators) — scale-invariant, no scaler.
* **LightGBM** (default hyperparameters) — scale-invariant, no scaler.

All three evaluated via Stratified 5-fold CV (ROC-AUC) on the training
split, then fit on the full training set and evaluated once on the
held-out test set:

| Model               | CV ROC-AUC | CV Std      | Test ROC-AUC |
| ------------------- | ---------- | ----------- | ------------ |
| Logistic Regression | 0.7489     | ±0.0024     | 0.7522       |
| Random Forest       | 0.7223     | ±0.0035     | 0.7246       |
| **LightGBM**        | **0.7617** | **±0.0015** | **0.7658**   |

LightGBM won on both CV and test by a clear margin and was selected for
tuning in `05`. Two notable warnings from this run: Logistic Regression
triggered `ConvergenceWarning` on every CV fold (lbfgs hit `max_iter=1000`
without converging — the 0.7489 CV AUC is from an unconverged model,
though it still generalised reasonably to test); Random Forest triggered
a `joblib` worker-stopped warning mid-CV run, indicating memory pressure
on the Kaggle session during the RF evaluation.

### 05 — Optuna Tuning

Tunes LightGBM's hyperparameters (`num_leaves`, `learning_rate`,
`n_estimators`, `min_child_samples`, `subsample`, `colsample_bytree`,
`reg_alpha`, `reg_lambda`) via Optuna's TPE sampler, 50 trials, using
**the identical train/test split as `04`** (same `random_state` and
`test_size` on the same source data reproduces it deterministically —
no need to persist split indices separately).

The search itself uses 3-fold CV rather than the full 5-fold — 50 trials
× 5 folds × LightGBM is an expensive way to buy marginal extra precision
during the search phase, when the loop only needs a stable enough signal
to _rank_ hyperparameter combinations against each other. The winning
parameters are then re-evaluated with the full 5-fold CV for the number
that actually gets reported.

Every trial is logged to MLflow (SQLite backend), and the Optuna study
itself uses persistent SQLite storage (`load_if_exists=True`) so a killed
or disconnected Kaggle session resumes from the last completed trial
rather than restarting from zero.

**Best parameters found:**
`num_leaves=25, learning_rate=0.0199, n_estimators=890,
min_child_samples=96, subsample=0.924, colsample_bytree=0.997,
reg_alpha=2.53, reg_lambda=0.00226`

The tuned pipeline is saved via `joblib` **before** MLflow's model-logging
step, so the artifact `06` depends on exists regardless of whether
MLflow's model-logging succeeds (MLflow's serializer can refuse to log
certain model internals — e.g. LightGBM's Booster class — as an
"untrusted type" safety measure; that's a tracking nicety, not something
that should block the actual saved model).

### 06 — Final Evaluation + SHAP

Loads `lightgbm_tuned.joblib` directly — no retraining, no Optuna. Two
deliverables:

1. **Bootstrap confidence interval** on test ROC-AUC: resamples the
   already-computed test predictions (not re-running the pipeline per
   resample) 1000 times with replacement, reporting the 95% percentile
   interval.
2. **SHAP TreeExplainer** analysis on the raw `LGBMClassifier` (fed the
   `ColumnTransformer`-transformed test features, with real post-encoding
   feature names via `get_feature_names_out()`), producing a beeswarm
   summary plot and a mean-|SHAP|-value bar chart for the top 5 features.

## Environment

- **Local (M4 Mac):** data prep, EDA, feature engineering — light,
  CPU-only steps. `ml-env`, Python 3.11.
- **Kaggle (Tesla T4 GPU):** baseline comparison, Optuna tuning, final
  eval — heavier or longer-running steps. `config.py` auto-detects the
  environment (`KAGGLE_KERNEL_RUN_TYPE` env var / `/kaggle/input`
  existence) and switches paths accordingly; no manual flags needed, and
  no separate local/Kaggle codebases to maintain.

## Project Structure

```
capstone-1-home-credit-default-risk/
├── 01_data_prep.py                        # load raw CSV, fix anomalies, drop junk columns, export clean parquet
├── 02_eda.py                              # visualize default patterns across key features
├── 03_feature_engineering.py             # create ratio/log features, export features parquet
├── 04_train_baseline_models.py           # compare LR, RF, LightGBM with stratified CV
├── 05_tune_best_model_optuna.py          # Bayesian HPO on LightGBM, log all trials to MLflow
├── 06_final_eval_shap.py                 # holdout evaluation, bootstrap CI, SHAP top-5
├── README.md                             # project overview, results, methodology
├── config.py                             # paths, column lists, model constants
├── data/
│   └── raw/
│       └── application_train.csv         # raw Kaggle dataset — 307K rows, 122 features
├── docs/
│   ├── problem_framing.md                # business context, success metrics, stakeholder framing
│   └── risk_report.md                    # deployment risks, data drift scenarios, failure modes
├── mlruns/                               # MLflow experiment store — auto-generated, do not edit
│   └── 1/
│       └── models/
│           └── m-4fe7c45d.../
│               └── artifacts/
│                   ├── MLmodel           # MLflow model metadata
│                   ├── conda.yaml        # conda env spec
│                   ├── model.skops       # serialized model (skops format)
│                   ├── python_env.yaml   # python env spec
│                   └── requirements.txt  # pip dependencies for this run
├── models/
│   └── lightgbm_tuned.joblib             # best tuned model saved with joblib
├── outputs/
│   ├── application_train_clean.parquet     # cleaned data from 01_data_prep.py
│   ├── application_train_features.parquet  # engineered features from 03_feature_engineering.py
│   ├── bootstrap_ci_results.csv            # 95% CI on final model metrics
│   ├── eda/
│   │   ├── 01_days_employed_anomaly_vs_default.png  # 365243 placeholder anomaly analysis
│   │   ├── 02_ext_source_missingness_vs_default.png # missingness as a signal
│   │   ├── 03_credit_income_ratio_vs_default.png    # engineered ratio vs target
│   │   └── 04_income_type_vs_default.png            # default rate by employment type
│   ├── model_comparison.csv              # CV ROC-AUC for all baseline models
│   ├── optuna/
│   │   ├── 01_optimization_history.png   # trial scores over time
│   │   ├── 02_param_importances.png      # which hyperparams mattered most
│   │   └── 03_parallel_coordinate.png    # param combinations vs score
│   ├── shap/
│   │   ├── 01_shap_summary_top5.png      # beeswarm plot — direction + magnitude
│   │   └── 02_shap_bar_top5.png          # mean |SHAP| bar chart
│   └── shap_top5_features.csv            # top-5 feature names + mean SHAP values
└── utils/
    ├── __init__.py                        # makes utils a package
    ├── data_utils.py                      # reusable loaders, cleaners, train/test splitters
    └── plot_utils.py                      # reusable chart helpers used across scripts


└── learning-journal/
    └── capstone-1.md                      # daily reflections: what was built, surprises, gaps
```

## Known Gaps / Not Yet Done

- Threshold-specific cost analysis (precision/recall trade-off at a
  real decision threshold, not just ROC-AUC) — see `risk_report.md`
  Section 5.
- Temporal/distributional drift evaluation — model has only been
  validated on a single historical snapshot.
- Exact quantification of `EXT_SOURCE_*` missingness population size
  and the zero/infinite-ratio edge case row count from `03`'s defensive
  check.

See `risk_report.md` for the full critical assessment.
