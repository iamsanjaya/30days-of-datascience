# Day 16 — Hyperparameter Tuning: Optuna + MLflow

**Date:** June 17, 2026
**Dataset:** AmesHousing.csv (https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## What I built today

A five-file hyperparameter tuning pipeline for XGBoost on Ames Housing:

- `config.py` — centralized search space, paths, and all constants
- `data_loader.py` — load, clean (drop >40% missing columns, median impute, one-hot encode), split into 2,344 train / 586 val rows across 254 features
- `optuna_tuner.py` — 55-trial TPE study with every trial logged as a nested MLflow child run
- `random_vs_bayesian.py` — out-of-box challenge: random vs Bayesian trials from the same seed, with exploration/exploitation labels on each Bayesian trial
- `plots.py` — four visualizations: optimization history, fANOVA parameter importance, parallel coordinates, random-vs-Bayesian comparison

Viewed all 55 trials in the MLflow dashboard. Every trial shows its full parameter set, CV RMSE, and RMSE standard deviation.

---

## Out-of-box challenge result

**Question asked:** Does Bayesian optimization (TPE) actually beat random search, and by how much?

**Results:**

- Random Search best RMSE: $22,299
- Bayesian (TPE) best RMSE: $22,045
- Winner: Bayesian (TPE) — by $254

**What this means:** Bayesian won, but the margin is so small that a different random seed could flip the result. This is the honest, empirical answer — Bayesian optimization is not automatically superior to random search. Its advantage compounds with more trials as the surrogate model accumulates signal. At lower trial counts, the two approaches are effectively equivalent, and random search has the advantage of being simpler and cheaper to reason about.

**Exploration vs exploitation:** Most Bayesian trials were labeled "exploration" (no improvement over running best). Only a few were "exploitation" (new best found). The surrogate model spends a large portion of its budget exploring before it can confidently exploit known good regions.

---

## What surprised me

1. **The best model is shallow.** `max_depth=3` with `n_estimators=829` and `learning_rate=0.037` — many shallow trees rather than few deep ones. Intuitively you might expect deeper trees to capture more signal, but boosting works by stacking weak learners. Depth adds variance; more trees with small steps add accuracy without overfitting.

2. **The CV-to-validation gap is $2,727 (~12%).** This is not alarming — it reflects the natural difference between averaging over CV folds and evaluating on a true holdout. But it is a reminder that CV RMSE is an optimistic estimate of real-world performance.

3. **$254 separates random from Bayesian.** This was the most instructive finding of the day. The out-of-box challenge could easily have produced a headline like "Bayesian is better" — but the actual margin forces a more honest conclusion: the sampler matters less than the trial budget.

---

## What I don't fully understand yet

- **TPE's internal math:** I understand the concept — model `l(x)` (distribution of good params) and `g(x)` (distribution of bad params) separately, sample where `l(x)/g(x)` is high — but I have not traced through how Optuna actually fits these distributions trial by trial.
- **fANOVA importance vs permutation importance:** both rank hyperparameters but via different mechanisms. fANOVA averages over the full search space; permutation importance measures change at the best point found. I am not yet clear on when one is more trustworthy than the other.
- **MLflow model registry:** today I used experiment tracking only. The model registry — versioning, staging, production promotion — is a separate workflow I have not touched yet.

---

## GitHub commit made: ✅

`Day 16 — Hyperparameter Tuning: Optuna + MLflow `

---

## Tomorrow's priority

Day 17 — End-to-end ML review + model card. Take the best model from Days 8–16 (the tuned XGBoost from today is the natural candidate), write a complete model card covering problem statement, dataset biases, evaluation with confidence intervals, known failure modes, and ethical considerations. Then run three robustness stress tests: distribution shift, 20% feature noise, and 10% label noise.
