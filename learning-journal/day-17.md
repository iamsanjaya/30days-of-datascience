# Day 17 — Model Card + Robustness Stress Test (Ames Housing)

**Date:** June 19, 2026

## What I built today

Rebuilt Day 15's XGBoost pipeline exactly as originally implemented, including rediscovery of missing persisted hyperparameters and validation of the actual training setup used previously. During reconstruction, it became clear that Day 14’s engineered features were never integrated into the Day 15 pipeline, which is a structural limitation of the existing system rather than a fix applied here.

Re-ran a 12-combination grid search using 5-fold cross-validation. The best configuration was:

- learning_rate = 0.1
- max_depth = 3
- n_estimators = 300

This produced:

- Test RMSE: 25,120
- RMSLE: 0.1086
- R²: 0.921
- RMSE CI: [19,966 → 30,435]

The final pipeline, test split, and baseline metrics were persisted for reproducibility.

---

## Stress tests performed

Three robustness evaluations were executed:

### 1. Distribution shift (Year Built split)

- Threshold: 1960
- Old homes (pre-1960):
  - RMSE: 18,204
  - RMSLE: 0.1342
- New homes (post-1960):
  - RMSE: 27,543
  - RMSLE: 0.0957

### 2. Feature noise (20% Gaussian noise)

- RMSE increased from 25,120 → 31,427
- Degradation: +25.1%

### 3. Label corruption (10%)

- RMSE increased from 25,120 → 52,944
- Degradation: +110.8%

---

## Key findings from robustness evaluation

Label corruption was the most damaging condition by a large margin. Even a 10% corruption rate more than doubled error, showing that the model is highly sensitive to incorrect supervision signals. This is expected in gradient-boosted tree models because corrupted labels directly distort the loss function and propagate incorrect gradient signals across all boosting iterations.

Feature noise had a significantly smaller effect. This suggests the model is relatively robust to moderate input perturbations, as tree-based ensembles can average out noisy feature splits and rely on redundant decision paths.

The distribution shift test revealed an important scale-dependent behavior. Older homes had lower absolute error (RMSE) but higher relative error (RMSLE), while newer homes showed the opposite pattern. This is driven by price scale differences rather than any inherent “preference” of the model.

---

## What surprised me

RMSE and RMSLE disagreed on which subgroup the model handled better. This is not a contradiction but a metric sensitivity difference:

- RMSE reflects absolute dollar error, so higher-priced homes (typically newer ones) dominate error magnitude.
- RMSLE compresses large values and emphasizes relative error, making performance on lower-priced homes (older ones) appear worse in proportional terms.

---

## What I misinterpreted initially

Label corruption is not simply “more harmful noise” compared to feature noise — it affects the optimization process differently.

- Feature noise: partially absorbed by tree ensembles through split redundancy and averaging across many weak learners.
- Label noise: directly corrupts the target signal used for gradient computation, causing systematic drift in all subsequent trees.

This makes label corruption structurally more damaging than feature perturbations in boosting models.

---

## Conclusion

The pipeline performs strongly on clean data (R² ≈ 0.92) but shows clear fragility under data quality degradation, especially label noise. The system is reasonably robust to input feature noise but highly sensitive to supervision quality, and its error behavior varies significantly depending on whether evaluation is absolute (RMSE) or relative (RMSLE).

## GitHub commit made: ✅

`Day 17 — Model Card + Robustness Stress Test`

## Tomorrow's priority:

Day 18 — build a 2-layer NN from scratch in NumPy, replicate it in Keras on MNIST, then visualize first-layer weights on real vs. randomly-shuffled labels to see what the network actually learns vs. memorizes.
