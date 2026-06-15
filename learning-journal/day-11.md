# Day 11 — June 13, 2026

## What I built today

### `01_xgboost_titanic.py` — XGBoost with Stratified K-Fold CV

Trained XGBoost against a Random Forest baseline on Titanic survival prediction.
Used Stratified 5-Fold CV with ROC-AUC as the primary metric and produced four
diagnostic plots: fold comparison bar chart, ROC curve, confusion matrix, and
precision-recall curve.

XGBoost CV ROC-AUC: 0.8803 ± 0.0248 vs Random Forest: 0.8723 ± 0.0249.
The margin (+0.008 AUC) is honest — Titanic is a well-explored dataset and the
signal XGBoost extracts incrementally over RF is real but modest.

Also added early stopping via `eval_set` on the held-out fold to prevent overfitting
across 300 estimators.

### `02_metric_comparison.py` — The Metric You Pick Changes Everything

Same model, same probability outputs — four different threshold-optimised operating
points. Actual results from my run:

| Optimised For | Threshold | Accuracy | F1     | Precision | Recall | FN (missed) | FP (false alarms) |
| ------------- | --------- | -------- | ------ | --------- | ------ | ----------- | ----------------- |
| Accuracy      | 0.355     | 0.8539   | 0.8116 | 0.8116    | 0.8116 | 13          | 13                |
| F1            | 0.355     | 0.8539   | 0.8116 | 0.8116    | 0.8116 | 13          | 13                |
| Precision     | 0.985     | 0.6573   | 0.2078 | 1.000     | 0.1159 | 61          | 0                 |
| Recall        | 0.010     | 0.3989   | 0.5633 | 0.392     | 1.000  | 0           | 107               |

These are not subtle variations. They represent completely different machine behaviour
produced by moving a single number.

---

## The out-of-box challenge result

The Titanic rescue framing forced me to think about costs, not scores. The model
that maximises Recall (threshold 0.010, accuracy 0.399) misses zero survivors.
The Precision model (threshold 0.985) leaves 61 behind. In a rescue scenario,
those 61 are not acceptable collateral — they are people.

The written business argument: a 44.38% drop in accuracy is a rational trade if the
domain cost of a false negative (a person's life) is categorically higher than the
domain cost of a false positive (a wasted rescue attempt). The optimal threshold is
not the one that maximises any metric in isolation — it is the one that minimises
total expected cost given domain knowledge of the cost asymmetry.

This is the kind of reasoning the roadmap described as "thinking like a domain expert,
not a Kaggler." The confusion matrices across all four operating points are in
`outputs/metric_comparison.png`.

---

## What surprised me

Two things from the actual output:

**1. Accuracy and F1 converged to the exact same threshold (0.355).** This is not
a bug — it is a property of this dataset. With Titanic's mild imbalance (~38%
positive class), both metrics are pulled toward the same operating point. On a
more skewed dataset, they diverge. Day 12 will test this: at 1:50 imbalance,
Accuracy and F1 will no longer agree. This is one concrete reason why class
imbalance changes everything about evaluation.

**2. Precision can reach exactly 1.0** at threshold 0.985 — but at the cost of
classifying only ~10% of actual survivors as survivors (Recall = 0.1159). A
perfect-precision model that stays silent on 89% of the positive class is not
a useful model. It is only "right" by being extremely conservative. This is
the precision trap — optimising a single metric without checking what it costs.

---

## What I don't fully understand yet

The shape of the precision-recall curve changes depending on dataset imbalance,
but I am not yet fully clear on _why_ the curve degrades differently from the ROC
curve under increasing imbalance. The ROC curve is relatively robust to imbalance
because it uses FPR (normalised by actual negatives). Precision-recall is not because
precision depends on the absolute count of FP relative to TP, which shifts as
class ratio shifts. I want to test this empirically on Day 12.

Also: XGBoost's `eval_metric="logloss"` inside the constructor — what exactly is
being optimised at each boosting round vs what Stratified K-Fold is scoring (AUC)?
These are different loss surfaces. I used them in the same script without fully
reconciling the relationship.

---

## GitHub commit made: ✅

`day-11: xgboost + metric selection philosophy`

## Tomorrow's priority

Day 12 — Class Imbalance. Manufacture imbalance ratios (1:2 through 1:100),
plot metric degradation curves, find the threshold where accuracy becomes useless
and where SMOTE stops helping. The Accuracy/F1 convergence finding from today
will be tested under deliberate imbalance — hypothesis: they diverge past 1:10.
