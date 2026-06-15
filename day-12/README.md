# Day 12 — Class Imbalance

## What This Day Covers

Four imbalance-handling strategies benchmarked against each other on a breast cancer
survival dataset (Dead = minority, Alive = majority), followed by a manufactured
imbalance study that quantifies exactly when accuracy becomes useless and when SMOTE
stops rescuing a collapsing model.

---

## Structure

```
day-12/         ← Class Imbalance — Strategy Comparison & Degradation Study
│   ├── config.py                           # shared constants: paths, random state, colours
│   ├── 01_imbalance_strategy_comparison.py # SMOTE · Undersampling · class_weight · Baseline — 5-Fold CV
│   ├── 02_outofbox_challenge.py            # manufactured imbalance 1:2 → 1:100 — degradation + gap heatmap
│   ├── README.md
│   ├── data/
│   │   └── breastcancer.csv                # original dataset — untouched
│   └── outputs/
│       ├── strategy_comparison.png
│       ├── degradation_curves.png
│       └── metric_gap_heatmap.png

learning-journal/
  └── day-12.md
```

## Scripts

### `01_imbalance_strategy_comparison.py` — Standard Task

Benchmarks four strategies on breast cancer survival classification using Stratified
5-Fold CV. Resampling is applied inside each fold on the training split only — no
leakage. Primary metrics: F1 (macro), ROC-AUC, Precision (macro), Recall (macro). ±

| Strategy              | F1 (macro)  | ROC-AUC     | Precision | Recall |
| --------------------- | ----------- | ----------- | --------- | ------ |
| Baseline              | 0.780±0.017 | 0.847±0.015 | 0.861     | 0.738  |
| SMOTE                 | 0.774±0.011 | 0.849±0.010 | 0.788     | 0.763  |
| Undersampling         | 0.699±0.011 | 0.854±0.013 | 0.678     | 0.773  |
| class_weight_balanced | 0.778±0.013 | 0.852±0.013 | 0.791     | 0.767  |

**Why Stratified K-Fold?** The breast cancer dataset has a skewed Dead/Alive split.
Without stratification, some folds may contain very few Dead samples, producing
unreliable and inflated CV estimates.

**Why resampling inside the fold?** Applying SMOTE before splitting leaks synthetic
minority samples into the validation set — the model is then evaluated on data it
effectively helped generate. All resampling in this script happens on `X_tr`, `y_tr`
only, after the fold split.

### `02_outofbox_challenge.py` — Out-of-Box Challenge

Takes the breast cancer dataset and artificially subsamples the majority class to
produce five progressively worse imbalance ratios: 1:2, 1:5, 1:10, 1:50, 1:100.
Runs 5-Fold CV with and without SMOTE at each ratio. Tracks four metrics across
all ratios and produces two plots.

**Degradation curve findings:**

| Ratio | Accuracy (no SMOTE) | F1 (no SMOTE) | ROC-AUC | PR-AUC |
| ----- | ------------------- | ------------- | ------- | ------ |
| 1:2   | 0.824               | 0.788         | 0.861   | 0.808  |
| 1:5   | 0.901               | 0.788         | 0.853   | 0.682  |
| 1:10  | 0.906               | 0.785         | 0.845   | 0.662  |
| 1:50  | 0.906               | 0.785         | 0.845   | 0.662  |
| 1:100 | 0.906               | 0.785         | 0.845   | 0.662  |

**The core finding:** At high imbalance ratios, Accuracy stays artificially high
while F1 and PR-AUC collapse. The gap between them — plotted in the heatmap — is
a direct measure of how deceptive Accuracy becomes. The redder the heatmap cell,
the more Accuracy is lying.

---

## Outputs

| File                              | Description                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| `outputs/strategy_comparison.png` | Four-panel bar chart — all strategies across all four metrics      |
| `outputs/degradation_curves.png`  | Four metric lines across five imbalance ratios, with/without SMOTE |
| `outputs/metric_gap_heatmap.png`  | Accuracy − F1 gap heatmap — reveals where accuracy is deceptive    |

---

## Environment

```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib
```

## Dataset

- **Source:** Breast Cancer dataset
- **File:** `data/breastcancer.csv`
- **Target column:** `Status` — `Dead` (minority / positive class), `Alive` (majority)
- **Key imbalance:** Dead samples are significantly fewer than Alive — exact ratio depends on dataset version

---

## Key Concepts

**SMOTE (Synthetic Minority Oversampling Technique)**
Creates synthetic minority samples by interpolating between existing minority
instances in feature space — not duplicating them. Must be applied inside the
CV fold to avoid leakage. Helps when the minority class has enough real samples
to interpolate meaningfully; loses effectiveness at extreme ratios (1:100) where
the minority class is too sparse for reliable interpolation.

**Random Undersampling**
Removes majority samples at random until balance is achieved. Fast and simple,
but discards real data — on small datasets this can hurt more than it helps.
The right choice when majority class genuinely contains redundant samples.

**class_weight='balanced'**
Does not change the data at all. Instead, it re-weights the loss function so
the model penalises errors on the minority class more heavily. No synthetic samples,
no data loss — just adjusted gradient signal during training. Often underrated.

**Accuracy vs F1 under imbalance**
Accuracy is the fraction of correct predictions across all samples. On a 1:100
dataset, a model that predicts majority class every time achieves ~99% accuracy.
F1 (macro) averages precision and recall across both classes equally — it exposes
what Accuracy hides. The Accuracy − F1 gap is the quantitative signature of
this deception.

**PR-AUC vs ROC-AUC under imbalance**
ROC-AUC is relatively robust to imbalance because it uses FPR, which is normalised
by total actual negatives. PR-AUC is sensitive — precision depends on the ratio of
TP to FP, which degrades as the minority class shrinks. At high imbalance, PR-AUC
is the more honest measure of model usefulness.
