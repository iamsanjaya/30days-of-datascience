# Day 15 — sklearn Pipelines & Data Leakage Quiz

**Dataset:** Ames Housing (2,930 rows × 82 cols · 79 model features · target: `SalePrice`)
**Phase:** Classical ML · Day 15 of 30

---

## Objective

Build a production-grade sklearn Pipeline on a real mixed-type dataset — numerical imputation, scaling, one-hot encoding, and XGBoost end-to-end. Then run GridSearchCV over the full pipeline. Out-of-box challenge: write and solve a 5-snippet data leakage quiz, annotating every leak at the exact line.

---

## File Structure

```
day-15/
├── config.py              # All constants, paths, feature lists, hyperparams
├── pipeline_factory.py    # build_preprocessor() + build_pipeline() — reusable
├── 01_load_explore.py     # Load, dtype audit, missing-value audit, train/test split
├── 02_pipeline.py         # Full pipeline + 5-fold CV + test set evaluation
├── 03_grid_search.py      # GridSearchCV over pipeline (12 combos × 5 folds = 60 fits)
├── 04_leakage_quiz.py     # 5-snippet leakage quiz — 3 leaky, 2 clean
├── data/
│   └── AmesHousing.csv
└── outputs/               # plots saved here
    ├── 01_missing_value audit.png
    ├── 01_target_distribution.png
    ├── 02_cv_fold_rmse.png
    ├── 02_feature_importance.png
    ├── 03_default _vs_tuned.png
    ├── 03_gridsearch_heatmap_lr01.png
    ├── 03_gridsearh_heatmap_lr005.png
    └── 04_leakage_inflation.png

learning-journal/
  └── day-15.md
```

---

## Dataset Audit

| Stat                 | Value                                    |
| -------------------- | ---------------------------------------- |
| Rows                 | 2,930                                    |
| Features (model)     | 79 (36 numeric + 43 categorical)         |
| Columns with missing | 27                                       |
| Highest missing      | Pool QC (99.6%) — absence, not error     |
| Target skew          | 1.744 — log-transform is worth exploring |

High-missing categoricals (Pool QC, Alley, Fence, Misc Feature) represent absent physical features — NaN means "no pool", not a data error. `most_frequent` imputation is used as a conservative pipeline default.

---

## Pipeline Architecture

```
ColumnTransformer
  ├── numeric (36 cols)
  │     SimpleImputer(median) → StandardScaler
  └── categorical (43 cols)
        SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
          │
          └── XGBRegressor
```

`handle_unknown='ignore'` ensures unseen categories at test time produce an all-zero OHE row rather than raising an error — critical for real-world robustness.

---

## Results

### 5-Fold CV (train set · `02_pipeline.py`)

| Metric        | Value                                                       |
| ------------- | ----------------------------------------------------------- |
| Val RMSE      | $23,194 ± $2,153                                            |
| Train RMSE    | $10,013 ± $166                                              |
| Val R²        | 0.9087 ± 0.0155                                             |
| Train-Val gap | $13,181 (overfit — shallower tree or regularisation needed) |

### Held-Out Test Set

| Metric                  | Value   |
| ----------------------- | ------- |
| Test RMSE               | $25,117 |
| Test R²                 | 0.9213  |
| RMSE as % of mean price | 13.2%   |

### GridSearchCV (12 combos · `03_grid_search.py`)

|                      | Value                                               |
| -------------------- | --------------------------------------------------- |
| Best params          | `lr=0.10, max_depth=3, n_estimators=300`            |
| Best CV RMSE         | $22,813                                             |
| Tuned test RMSE      | $26,090                                             |
| vs default test RMSE | $25,117                                             |
| Difference           | −$972 (tuned was slightly worse on this test split) |

The CV-best config didn't improve the held-out score. This is expected on a single test split of 586 rows — CV RMSE is the more reliable signal; test-set variance is high at this sample size.

### Top 5 Features (XGBoost gain)

| Feature         | Importance |
| --------------- | ---------- |
| Exter Qual_TA   | 0.648      |
| Overall Qual    | 0.054      |
| Garage Cars     | 0.045      |
| Kitchen Qual_Ex | 0.030      |
| Bldg Type_1Fam  | 0.019      |

`Exter Qual_TA` dominating at 64.8% gain is a signal to investigate: it may indicate the OHE dummy is acting as a proxy for the ordinal scale and the model is over-relying on one binary split.

---

## Out-of-Box Challenge — Leakage Quiz

5 snippets, all using Ames Housing data.

| Snippet | Verdict | Leak Type | Root Cause                                         |
| ------- | ------- | --------- | -------------------------------------------------- |
| 1       | LEAKY   | TYPE-A    | `StandardScaler.fit_transform(X_all)` before split |
| 2       | CLEAN   | —         | `Pipeline.fit(X_train)` — correct pattern          |
| 3       | LEAKY   | TYPE-B    | Target encoding mean computed on full df           |
| 4       | LEAKY   | TYPE-C    | `shift(-1)` encodes next sale (future info)        |
| 5       | LEAKY   | TYPE-D    | Pre-scaled X passed to `cross_val_score`           |

**Measured inflation from leakage:**

- Snippet 1: +0.000000 R² (Ames numeric features are well-behaved; inflation is dataset-dependent — on skewed or small data this is significant)
- Snippet 3: +0.0079 R² from target encoding leak
- Snippet 5: +0.000001 R² (Ridge on linear data; non-linear models show larger gaps)

The near-zero inflation on Snippets 1 and 5 does not mean leakage is harmless — it means Ames Housing numeric features happen to have mild outliers. On financial or medical data with skewed distributions, TYPE-A leakage routinely inflates reported performance by 2–5%.

---

## Key Concepts

**Why Pipeline prevents leakage:** `Pipeline.fit(X_train)` applies each transformer's `fit_transform` only to the training rows. `Pipeline.predict(X_test)` applies only `transform` — the fitted parameters (scaler mean, imputer fill value) never re-estimate from test rows.

**GridSearchCV over a pipeline:** Param names use `step__param` syntax (`model__max_depth`). This forces the grid search to tune through the pipeline abstraction — every candidate config gets a fresh preprocessor fit per fold.

**Leakage taxonomy used:**

- TYPE-A: preprocessing fit before split
- TYPE-B: target statistics computed on full data
- TYPE-C: future information encoded in features
- TYPE-D: preprocessing outside pipeline inside CV

---

## How to Run

```bash
# from day-15/ directory
python 01_load_explore.py   # data audit
python 02_pipeline.py       # pipeline + CV + test evaluation
python 03_grid_search.py    # hyperparameter search
python 04_leakage_quiz.py   # leakage quiz with printed verdicts
```

Requires: `pandas numpy scikit-learn xgboost`
