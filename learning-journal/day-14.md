# Day 14 — June 16, 2026

**Dataset:** AmesHousing.csv (https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
**Scripts:** `01_feature_engineering.py`, `02_anti_feature_ablation.py`

---

## What I built today

**Script 01 — Feature Engineering:**

- Engineered feature set added on top of raw housing attributes:
  - TotalSF
  - Age
  - RemodAge
  - HasPool
  - HasGarage
  - log-transformed skewed variables (Lot Area, Gr Liv Area, Total Bsmt SF, 1st Flr SF, Garage Area, Mas Vnr Area)
- Baseline vs engineered model comparison using XGBoost with 5-fold CV
- Slight but consistent improvement after feature engineering
- Output dataset persisted as data/processed/engineered.csv

**Script 02 — Anti-Feature Ablation:**

- Permutation importance computed over full feature space
- Greedy backward elimination starting from least important features
- Iterative CV evaluation until RMSLE exceeded 95% retention boundary
- Ablation log recorded per step (features_remaining, rmsle_mean)
- Ablation curve generated showing degradation vs feature removal
- Feature importance exported (permutation-based ranking)

---

## Results

| Metric                                     | Value   |
| ------------------------------------------ | ------- |
| Baseline RMSLE (no engineering)            | 0.12099 |
| Engineered RMSLE                           | 0.11916 |
| Improvement %                              | ~1.51%  |
| Full model feature count                   | ~70+    |
| Minimal feature set count                  | ~67     |
| Engineered features that survived ablation | 2       |
| Engineered features eliminated             | 3       |

---

## The out-of-box challenge result

### Which engineered features survived the cut

- TotalSF
- HasGarage

### Which were eliminated

- Age
- RemodAge
- HasPool

### Observation

Time-based engineered features were least stable under ablation. TotalSF remained the strongest engineered signal.

---

## What surprised me

- Model performance remained stable even after many feature removals
- A small subset of raw features dominated prediction
- Some intuitive features (Age, RemodAge) had low marginal value
- Feature interactions likely matter more than individual feature strength

---

## What I don't fully understand yet

- Permutation importance is more reliable than tree-based importance, but still sensitive to correlation between features.
- Greedy ablation may miss optimal subsets due to interaction effects.
- 95% retention is a heuristic, not a rule.

---

## Key concept reinforced

- Feature usefulness is defined by how much independent signal it adds under evaluation pressure, not how it is engineered.

---

## GitHub commit made: ✅

`day-14: feature engineering + anti-feature ablation study`

---

## Tomorrow's priority (Day 15)

sklearn Pipelines — wrapping imputation + encoding + scaling + XGBoost into a single `Pipeline` object with `GridSearchCV` over the full pipeline. The data leakage quiz: write 5 code snippets, annotate where leakage occurs.
