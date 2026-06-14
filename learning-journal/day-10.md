# Day 10 — June 12, 2026

## What I built today

Trained a Decision Tree (depth 4) on the IBM Telco Customer Churn dataset (7,043 rows,
19 features) and compared it against a Random Forest (200 trees, depth 8) using 5-fold
Stratified CV. Built a full pipeline across three scripts: `01_` for the standard task
(train, visualize, CV compare), `02_` for the out-of-box challenge (decision path tracing
and leaf flattening analysis), and `03_` for extended RF analysis (OOB error curve and
feature importance stability across 10 random seeds). Produced 7 PNG outputs total.

## The out-of-box challenge result

Used `decision_path()` to manually trace how each of 10 sample rows flows through the
tree — printing the exact feature value and threshold at every node until the leaf. Then
found the most populated leaf (the one the tree funnels the most customers into) and
identified the most dissimilar pair of customers inside it using L1 pairwise distance.
Both customers received the same `P(Churn)` score despite differing significantly on
`tenure` and `MonthlyCharges`. This is the leaf flattening problem in practice: a tree
with only 4 levels runs out of splits before it can distinguish fine-grained differences
within a segment. The DT vs RF probability distribution plot made this visually obvious —
DT probabilities spike at discrete values (one per leaf), while RF probabilities form a
smooth continuous distribution averaged across 200 trees.

## What surprised me

Two things. First, that `Contract` dominates the single tree so completely — roughly 59%
of the tree's total importance collapses into that one root split. The whole tree is
essentially asking "month-to-month or not?" before it does anything else. Second, that OOB
error stabilises well before 200 trees on this dataset. The curve flattens around 75–100
estimators, which means the default `n_estimators=100` in sklearn is probably fine here
and the extra 100 trees in my config are buying almost nothing. That is a useful thing to
check empirically rather than assume.

## What I don't fully understand yet

Feature importance in Random Forest is measured as mean decrease in impurity (MDI) across
all trees. I understand what this means in principle but I am not sure how to interpret
it when two features are correlated — for example, `tenure` and `TotalCharges` are almost
certainly correlated (longer tenure → higher total charges), so does the forest split
importance between them arbitrarily? I have seen references to permutation importance as a
more reliable alternative for correlated features but have not implemented it yet. That
feels like the right thing to explore before Day 14's feature engineering work.

## GitHub commit made: ✅

`day-10: decision tree path analysis + random forest comparison`

## Tomorrow's priority

Day 11 — XGBoost on Titanic with Stratified K-Fold CV, ROC-AUC as primary metric, early
stopping, and proper test holdout. Goal: beat Random Forest by at least 1% AUC. Also need
to explore the "metric you pick changes everything" out-of-box challenge — optimising for
Accuracy vs F1 vs Precision vs Recall and seeing which model "wins" changes depending on
the metric chosen.
