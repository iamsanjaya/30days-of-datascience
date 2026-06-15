# Day 12 — June 14, 2026

## What I built today

### `01_imbalance_strategy_comparison.py` — Strategy Comparison

Benchmarked four class imbalance strategies — Baseline, SMOTE, Random Undersampling,
and class_weight='balanced' — on a breast cancer survival dataset using Stratified
5-Fold CV with Random Forest. All resampling was applied strictly inside each
training fold to prevent leakage. Metrics tracked: F1 (macro), ROC-AUC, Precision
(macro), Recall (macro). Output: a four-panel bar chart with error bars across all
strategies.

### `02_outofbox_challenge.py` — Manufactured Imbalance Study

Artificially subsampled the majority class (Alive) to produce five imbalance ratios:
1:2, 1:5, 1:10, 1:50, 1:100. Ran 5-Fold CV with and without SMOTE at each ratio.
Tracked Accuracy, F1, ROC-AUC, and PR-AUC across all five points. Output: a
four-panel degradation curve plot and an Accuracy − F1 gap heatmap that shows
exactly where and how much Accuracy starts lying.

---

## The out-of-box challenge result

The degradation curves confirmed what Day 11 predicted: Accuracy and F1 diverge
under imbalance, and the divergence is not gradual — it accelerates past a threshold.
At 1:10, the gap is noticeable. At 1:50 and 1:100, Accuracy remains high while F1
has collapsed to near-random levels. The heatmap makes this viscerally clear — the
gap cells at 1:100 are dark red while the 1:2 cells are nearly white.

SMOTE helps — but not uniformly. Its largest F1 gain appears at mid-range ratios
(around 1:10 to 1:50), where the minority class still has enough real samples to
interpolate meaningfully. At 1:100, the minority class is so sparse that SMOTE's
synthetic samples become unreliable — they are generated from a handful of real
points and do not represent the true minority distribution. The gain shrinks or
disappears at extreme ratios. SMOTE is not a solution to extreme imbalance; it
is a solution to moderate imbalance.

The written finding: there is a "usability threshold" — a ratio beyond which no
standard resampling strategy restores model usefulness on the minority class.
Where exactly that threshold falls depends on dataset size and minority class
separability, but on this dataset it became visible around 1:50.

---

## What surprised me

**1. class_weight='balanced' competed with SMOTE at moderate imbalance.** I expected
SMOTE to win clearly at all ratios because it physically adds minority samples.
Instead, at 1:2 and 1:5, adjusting the loss weights produced comparable F1 to
SMOTE without generating any synthetic data. This matters practically — class_weight
is one line of code with zero risk of introducing synthetic noise. It is underrated.

**2. PR-AUC degraded faster and earlier than ROC-AUC.** ROC-AUC held up reasonably
well even at 1:50 before visibly declining. PR-AUC started dropping much earlier
and more steeply. This is the empirical proof of what Day 11's journal noted
theoretically: ROC-AUC is normalised by actual negatives (FPR), making it
relatively imbalance-resistant. Precision is not — it depends on the raw ratio
of TP to FP, which gets worse as minority samples disappear from the data. At high
imbalance, PR-AUC is the more honest performance measure.

---

## What I don't fully understand yet

SMOTE interpolates between existing minority samples in feature space. At extreme
imbalance (1:100), there are very few minority anchor points, so the synthetic
samples cluster tightly around the few real ones. But I am not fully clear on
whether this tight clustering is uniformly bad, or whether it depends on the
geometry of the minority class in feature space. If the minority class is genuinely
compact (all Dead patients share similar feature patterns), tight synthetic samples
might actually be fine. If the minority class is spread out and heterogeneous,
they would be misleading. I want to understand how to test which case applies
before deciding whether SMOTE is appropriate at a given imbalance level.

Also: undersampling performed worse than expected at moderate ratios. I assumed
discarding majority samples would be more harmful on larger datasets than small
ones — but I do not have a precise mental model for when undersampling becomes
preferable to oversampling. The crossover point probably depends on dataset size
relative to model complexity, but I have not worked through the reasoning
rigorously yet.

---

## GitHub commit made: ✅

`day-12: class imbalance study + degradation analysis`

## Tomorrow's priority

Day 13 — Clustering & Dimensionality Reduction. K-Means with elbow + silhouette
selection, PCA and t-SNE for 2D cluster visualisation, customer profiling per
cluster. The out-of-box challenge runs K=3, K=5, and K=8 on the same dataset and
asks whether different K values tell fundamentally different stories or just merged
versions of each other — testing whether clustering is a lens or a truth.
