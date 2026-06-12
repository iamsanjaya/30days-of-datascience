# Day 09 — Logistic Regression + Threshold as a Medical Ethics Decision

## Standard Task

Binary classification on a disease screening dataset using Logistic Regression.

**Plots produced:**

- ROC curve with AUC
- Confusion matrix at default threshold
- Precision-Recall curve

**Results:** ROC-AUC = 0.9531

## Out-of-Box Challenge

Reframed threshold selection as a cost-based decision for a hospital setting.

**Key insight:** The threshold that minimizes errors (F1-optimal) is not the same as the threshold that minimizes _harm_. When a false negative (missed disease) costs 10× more than a false positive (unnecessary referral), the optimal threshold shifts from 0.50 → 0.35, reducing total weighted cost by ~90%.

See [`decision_document.md`](./decision_document.md) for the full non-technical write-up.

## Scripts

| Script                     | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| `01_train_logistic.py`     | Train model, ROC + confusion matrix + PR curve      |
| `02_threshold_analysis.py` | Sweep thresholds, show precision/recall/F1 tradeoff |
| `03_cost_framework.py`     | Cost-weighted optimal threshold selection           |

## Files

- `data/heart.csv` — Cleveland Heart Disease dataset (processed, UCI), 297 rows after dropna, binary target
- `decision_document.md` — non-technical threshold decision document
- `outputs/` — gitignored
