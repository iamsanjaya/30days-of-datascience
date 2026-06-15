# Day 11 — XGBoost & Evaluation Metrics

## What This Day Covers

XGBoost with Stratified K-Fold cross-validation on Titanic survival prediction,
followed by a threshold-tuning study demonstrating that the "best" model is entirely
defined by which metric you choose to optimise.

---

## Structure

```
day-11/         ← XGBoost & Evaluation - Metric Selection Philosophy
│   ├── config.py                       # shared constants: paths, random state, model params, colours
│   ├── 01_xgboost_titanic.py           # XGBoost vs Random Forest — Stratified 5-Fold CV + diagnostic plots
│   ├── 02_metric_comparison.py         # threshold tuning per metric — rescue scenario ethics analysis
│   ├── README.md
│   ├── data/
│   │   ├── raw/
│   │   │   └── titanic_uncleaned.csv   # original Kaggle Titanic — untouched
│   │   └── processed/
│   │       └── titanic_cleaned.csv     # imputed, encoded, leakage columns dropped
│   └── outputs/
│       ├── confusion_matrix.png
│       ├── cv_scores.png
│       ├── metric_comparison.png
│       ├── precision_recall_curve.png
│       ├── roc_curve.png
│       └── threshold_analysis.png

learning-journal/
  └── day-11.md


```

## Scripts

### `01_xgboost_titanic.py` — Standard Task

Trains XGBoost against a Random Forest baseline using Stratified 5-Fold CV with
ROC-AUC as the primary metric. Produces four diagnostic plots.

Key results:

| Model         | CV ROC-AUC (5-fold) | Std Dev |
| ------------- | ------------------- | ------- |
| XGBoost       | 0.8798              | ±0.0245 |
| Random Forest | 0.8731              | ±0.0220 |

XGBoost wins by +0.0067 AUC across all five folds. The margin is modest because
Titanic is a well-explored dataset without strong non-linear signal that boosting
exploits dramatically — this is an honest result.

**Why Stratified K-Fold?** Titanic has a 38/62 class split. Without stratification,
fold-level class ratios vary randomly, inflating CV variance and producing unreliable
score estimates.

### `02_metric_comparison.py` — Out-of-Box Challenge

Same model, same probability scores — four different operating decisions.

| Optimised For | Threshold | Accuracy | F1     | Precision | Recall | FN (missed survivors) | FP (false alarms) |
| ------------- | --------- | -------- | ------ | --------- | ------ | --------------------- | ----------------- |
| Accuracy      | 0.355     | 0.8539   | 0.8116 | 0.8116    | 0.8116 | 13                    | 13                |
| F1            | 0.355     | 0.8539   | 0.8116 | 0.8116    | 0.8116 | 13                    | 13                |
| Precision     | 0.985     | 0.6573   | 0.2078 | 1.0000    | 0.1159 | 61                    | 0                 |
| Recall        | 0.010     | 0.3989   | 0.5633 | 0.3920    | 1.0000 | 0                     | 107               |

**The rescue scenario argument:** In a real-time rescue context, a False Negative
(FN) means a living person was classified as dead and not rescued. A False Positive
(FP) means a rescue attempt was made for someone already dead — a wasted trip.
These are not equivalent costs. FN = human life. FP = resource. Therefore, a
domain-rational operator would use the Recall-optimised threshold (0.010), accepting
44% accuracy to achieve zero missed survivors.

The model is not just its architecture. It is also its operating point.

---

## Outputs

| File                                 | Description                                            |
| ------------------------------------ | ------------------------------------------------------ |
| `outputs/cv_scores.png`              | XGBoost vs Random Forest — per-fold ROC-AUC comparison |
| `outputs/roc_curve.png`              | ROC curve with AUC for held-out test fold              |
| `outputs/confusion_matrix.png`       | Confusion matrix at default 0.50 threshold             |
| `outputs/precision_recall_curve.png` | Precision-Recall curve with average precision          |
| `outputs/threshold_analysis.png`     | All four metrics plotted across threshold range 0–1    |
| `outputs/metric_comparison.png`      | Confusion matrices at each metric's optimal threshold  |

---

## Environment

```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

## Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **File:** `data/titanic_uncleaned.csv`
- **Shape:** 891 rows × 12 columns
- **Key missing values:** Cabin (77.1%), Age (19.9%), Embarked (0.2%)

---

## Key Concepts

**XGBoost vs Random Forest**
Random Forest builds trees in parallel, each on a bootstrap sample. XGBoost builds
trees sequentially — each tree fits the gradient of the loss on the residuals of
all previous trees. This sequential correction is why XGBoost tends to outperform
RF on structured data with sufficient signal.

**Stratified K-Fold**
Guarantees each fold preserves the original class ratio. Essential for imbalanced
classification — without it, CV scores have high variance and are untrustworthy.

**ROC-AUC as primary metric**
Threshold-independent. Represents the probability that the model ranks a random
positive above a random negative. Robust for model comparison regardless of
operating threshold chosen downstream.

**Threshold tuning**
The model outputs a probability. The threshold converts it to a class label.
Moving the threshold changes the precision-recall trade-off. There is no single
"correct" threshold — it is a business decision, not a mathematical one.
