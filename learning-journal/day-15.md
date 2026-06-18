# Day 15 — sklearn Pipelines + Data Leakage Quiz

**Date:** June 17, 2026
**Dataset:** Ames Housing (2,930 rows · 79 features · target: SalePrice)

---

## What I built today

A production-grade sklearn Pipeline for the Ames Housing dataset — 36 numeric features through median imputation and StandardScaler, 43 categorical features through most_frequent imputation and OneHotEncoder, all feeding into XGBoost. Ran 5-fold CV on the training set, evaluated on a held-out test set, then ran GridSearchCV over 12 hyperparameter combinations (60 total fits) through the full pipeline using the `model__param` double-underscore syntax. For the out-of-box challenge, wrote and ran a 5-snippet data leakage quiz using only Ames Housing data — 3 leaky (TYPE-A scaler leak, TYPE-B target encoding leak, TYPE-C future-shift leak, TYPE-D CV contamination), 2 clean — with exact leak lines annotated and printed verdicts.

---

## The out-of-box challenge result

Ran all 5 leakage snippets with measured R² comparisons:

- **Snippet 1 (TYPE-A — scaler before split):** Inflation = +0.000000. The Ames numeric features are well-behaved enough that the scaler's mean/std barely shifts when test rows are included. This does not mean the leak is safe — on skewed financial data this regularly inflates scores by 2–5%. The principle stands; the dataset happened to be forgiving.
- **Snippet 3 (TYPE-B — target encoding on full df):** Inflation = +0.0079 R². Small but measurable. Target encoding leaks test-row sale prices back into training — the model indirectly sees the answer.
- **Snippet 5 (TYPE-D — pre-scaled X into cross_val_score):** Inflation = +0.000001. Ridge on mostly-linear numeric features — again, the leak is real but the dataset absorbs it. Non-linear models on high-cardinality data show much larger gaps.

The deeper takeaway: measuring the inflation only works when you have a clean version to compare against. In a real project you often don't — the leaky score is the only number you see, and it looks good. That is exactly why leakage is dangerous.

---

## What surprised me

Two things:

**1. The GridSearch best config performed worse on the held-out test than the default config.** CV-best was `lr=0.10, max_depth=3, n_estimators=300` with CV RMSE $22,813. But on the 586-row test set it scored $26,090 vs the default's $25,117. A $973 regression. This isn't a bug — it's variance. The test set is small (586 rows), and a single split is a noisy estimator of true generalisation error. CV RMSE is the signal; single test-set comparison is a coin flip at this sample size. The correct interpretation: both configs are within CV noise range of each other, and I'd need repeated hold-out splits or a larger test set to call a winner.

**2. `Exter Qual_TA` captured 64.8% of XGBoost's total gain.** That is a near-complete dominance from a single OHE dummy. This almost certainly means the ordinal quality scale (Ex/Gd/TA/Fa/Po) is being handled suboptimally by OHE — treating it as unordered categoricals when it has a clear ordering. Ordinal encoding or treating Overall Qual as a continuous proxy would be the right domain fix. It's on the list for Day 17's model card.

---

## What I don't fully understand yet

Why the Train-Val RMSE gap is $13,181 ($10,013 train vs $23,194 val) when `max_depth=4` and XGBoost has implicit regularisation via `subsample=0.8` and `colsample_bytree=0.8`. The model is clearly memorising training structure that doesn't generalise. Depth-3 in the GridSearch helped close the gap — CV RMSE dropped to $22,813 — but the training RMSE on those folds was presumably much lower. I want to understand at what `max_depth` the train-val gap closes and whether `min_child_weight` or `gamma` (not in today's grid) would be the right lever to pull. That's an Optuna Day 16 question.

---

## GitHub commit made: ✅

`day-15: sklearn pipeline + grid search + leakage quiz on ames housing`

## Tomorrow's priority

Day 16 — Optuna hyperparameter tuning with 50+ trials, MLflow experiment logging, and the exploration-exploitation visualisation of the optimisation history.
