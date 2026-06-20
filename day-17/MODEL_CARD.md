# Model Card — Ames Housing Price Predictor

> Subject of this card: the XGBoost regression pipeline built in `day-15/`,
> rebuilt and frozen in `day-17/01_rebuild_best_pipeline.py`. Hyperparameters
> were rediscovered via grid search since Day 15 never persisted them.
> Fill in the `[[ ]]` placeholders from `outputs/baseline_metrics.json`,
> `outputs/best_params.json`, and `ROBUSTNESS_REPORT.md` after running scripts
> 01–05.

---

## 1. Problem Statement

Predict the sale price of a residential property in Ames, Iowa, from its
physical and qualitative characteristics (size, condition, location,
construction details) recorded at time of sale. Intended use: a regression
benchmark for comparing feature-engineering and pipeline-tuning techniques
across Days 14–16 of this program — **not** a production valuation tool.
Anyone using this model to actually price a house is using it outside its
validated scope (see §6).

## 2. Dataset Description & Known Biases

- **Source:** Ames Housing dataset, 2930 rows, 82 raw columns, Kaggle.
- **Geography:** Single city (Ames, Iowa) and surrounding area. The model
  has no exposure to other housing markets, climates, building codes, or
  price regimes — generalization outside Ames is untested by construction,
  not just untested by omission.
- **Time range:** Sales span multiple decades (`Year Built` and `Yr Sold`
  cover a wide range — see `[[min_year_built]]`–`[[max_year_built]]`).
  Older and newer homes are pooled into one training set; §6 and the
  distribution-shift stress test address whether the model treats them
  evenly.
- **Feature set used:** the 36 raw numeric + 43 raw categorical columns
  listed in `day-15/config.py` (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`).
  `Order` and `PID` (row identifiers) are dropped.
- **Known sampling bias:** Ames is a university town (Iowa State University)
  — housing stock and buyer demographics likely skew toward that population
  relative to a typical American city. Not corrected for.

## 3. Model Architecture & Training Details

- **Pipeline:** `ColumnTransformer` (median imputation + `StandardScaler` on
  numeric; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
  on categorical) → `XGBRegressor`.
- **Hyperparameter search:** `GridSearchCV`, 5-fold CV, scoring =
  `neg_root_mean_squared_error`, grid = `{n_estimators: [100, 300],
max_depth: [3, 4, 6], learning_rate: [0.05, 0.10]}` (12 combinations,
  60 total fits).
- **Best parameters found:** `[[ best_params from outputs/best_params.json ]]`
- **Train/test split:** 80/20, `random_state=42`, no stratification
  (continuous target).
- **Best CV score (neg RMSE):** `[[ cv_best_score_neg_rmse ]]`

## 4. Evaluation Metrics (with Confidence Interval)

All metrics computed on the untouched 20% test split.

| Metric                          | Value                                             |
| ------------------------------- | ------------------------------------------------- |
| RMSE                            | `[[ rmse ]]`                                      |
| RMSE 95% CI (bootstrap, n=1000) | `[[ rmse_ci.ci_lower ]] – [[ rmse_ci.ci_upper ]]` |
| RMSLE                           | `[[ rmsle ]]`                                     |
| R²                              | `[[ r2 ]]`                                        |
| n_test                          | `[[ n_test ]]`                                    |

## 5. Known Failure Modes

1. **Day 14's engineered features were never integrated into this pipeline.**
   This isn't a hypothetical limitation — it's a documented fact about this
   specific model's history. `TotalSF`, `Age`, `RemodAge`, `HasPool`,
   `HasGarage`, and the log-transformed columns from `day-14/utils/
feature_builder.py` exist in the repo but were never folded into Day 15's
   `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES`. The model is therefore working
   with more raw, less-informative signal than it could be.
2. **Robustness stress test results** (full detail in `ROBUSTNESS_REPORT.md`):
   - Distribution shift (old vs. new homes within the held-out test set):
     `[[ summarize old vs new RMSE gap ]]`
   - 20% feature noise: `[[ rmse_degradation_pct from noise_injection_results.json ]]`
   - 10% label corruption: `[[ rmse_degradation_pct from label_corruption_results.json ]]`
3. **Geographic/temporal narrowness** — see §2. No evidence this model
   transfers outside Ames or outside its training time window.
4. **Hyperparameters were re-discovered, not validated independently** — the
   same 5-fold CV that selected `[[ best_params ]]` is the same CV reported
   in §4, so there's no fully independent confirmation the grid search
   didn't overfit to this particular fold split.

## 6. Ethical Considerations

- **Not a valuation or lending tool.** Real-estate price models, when used
  in lending or insurance decisions, can encode and launder historical
  geographic bias (e.g. via `Neighborhood`) into seemingly "objective"
  numeric scores. This model was built as a learning exercise and has none
  of the fairness auditing that would be required before any such use.
- **Neighborhood as a categorical feature** is a strong predictor in most
  Ames Housing analyses and is also a well-known proxy for historically
  discriminatory housing patterns in U.S. real estate generally. No fairness
  analysis across `Neighborhood` groups was performed for this card.
- **No demographic or protected-class data exists in this dataset**, which
  limits direct discrimination analysis but does not guarantee the absence
  of indirect proxies (location, lot size, construction era can all
  correlate with historical redlining patterns in many U.S. cities,
  Ames included or not — this was not specifically verified here).
