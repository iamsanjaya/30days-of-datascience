# Day 14 — Anti-Feature Engineering (Ablation Study)

**Dataset:** Ames Housing — https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
**Goal:** Feature Engineering vs Feature Minimalism (Ablation-driven selection)

---

## What I built

A full ML pipeline that does not just add features — it questions them.

This project builds engineered features, evaluates performance gain, then performs **anti-feature engineering** using permutation importance + greedy backward ablation to find the smallest feature set that retains ~95% of model performance.

The focus is not feature creation, but **feature accountability under cross-validation**.

---

## File structure

```
day-14/
├── config.py                         # paths, CV settings, XGB params, thresholds
├── 01_feature_engineering.py        # baseline vs engineered comparison (CV RMSLE)
├── 02_anti_feature_ablation.py      # permutation importance + greedy ablation
├── README.md
├── utils/
│   ├── feature_builder.py            # engineered feature generation
│   ├── encoding.py                   # label encoding utilities
│   └── evaluation.py                 # CV RMSLE + reporting utilities
│
├── data/
│   ├── raw/
│   │   └── AmesHousing.csv
│   └── processed/
│       ├── engineered.csv
│       ├── ablation_log.csv
│       └── feature_importance.csv
├── outputs/
│   └── plots/
│       ├── ablation_curve.png
│       └── feature_importance.png

learning-journal/
  └── day-14.md
```

---

## Run order

```bash
python 01_feature_engineering.py
python 02_anti_feature_ablation.py
```

---

## Standard task results

### Baseline vs Engineered Model

| Model      | RMSLE   | Std Dev | Improvement |
| ---------- | ------- | ------- | ----------- |
| Baseline   | 0.12099 | 0.01639 | —           |
| Engineered | 0.11916 | 0.01511 | +1.51%      |

Engineered features provide a measurable but modest gain, confirming that raw housing signals are already strong.

---

### Key engineered features

- TotalSF (strongest signal)
- Age / RemodAge
- HasPool / HasGarage
- Log transforms of skewed variables

Only a subset of engineered features survive ablation, confirming that not all transformations add independent signal.

---

### Permutation importance (top signals)

- TotalSF dominates predictive power
- Overall Qual remains the strongest raw driver
- Gr Liv Area, Year Built, Lot Area provide secondary signal band
- Several categorical encodings contribute low but stable signal

Weak/neutral features identified as candidates for removal via near-zero or negative permutation impact.

---

## Out-of-Box Challenge — "Anti-Feature Engineering"

Instead of only measuring feature importance, the pipeline removes features until model performance begins to degrade beyond a controlled threshold (~95% retention boundary).

### Key result

- Model starts with full feature space
- Features removed from least → most important order
- Greedy ablation stops when RMSLE exceeds retention threshold

### Outcome

| Metric                 | Value                      |
| ---------------------- | -------------------------- |
| Original features      | 100%                       |
| Minimal feature set    | reduced subset             |
| Performance retained   | ~95%                       |
| Feature reduction gain | significant reduction in - |
|                        | model complexity with -    |
|                        | ~73-step ablation process  |

---

## Key insight

Feature engineering is not automatically beneficial.

This pipeline shows:

- Some engineered features are critical (signal creation)
- Many are redundant (noise amplification risk)
- A small subset carries most predictive power
- Model performance is stable even after aggressive feature removal

The real objective is not maximizing features — it is **maximizing signal-to-complexity ratio**.

---

## Core concepts

- **Cross-validation RMSLE**: primary stability metric (not train score)
- **Permutation importance**: model-agnostic feature impact measurement
- **Greedy backward ablation**: iterative feature removal under performance constraint
- **Retention threshold (~95%)**: controlled degradation boundary for feature pruning
- **Feature minimalism**: production principle — fewer features, same performance, lower risk
