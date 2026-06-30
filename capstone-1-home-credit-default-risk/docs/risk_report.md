# Risk Report — Home Credit Default Risk Model

**Project:** Capstone 1 — Structured Data ML
**Model:** LightGBM (Optuna-tuned), full sklearn Pipeline
**Author's note:** This report argues _against_ uncritical deployment of this
model, per the Day 24–25 "Anti-Pitch" exercise. The goal is to surface where
this model could fail in production, not to restate why it works.

---

## 1. What This Model Does

Predicts `TARGET` (loan default risk, binary) for Home Credit loan
applicants, using application-level features (income, credit amount,
demographics, external bureau scores, housing/asset details) plus four
engineered features (`AGE_YEARS`, `CREDIT_INCOME_RATIO`,
`ANNUITY_INCOME_RATIO`, `CREDIT_TERM_RATIO`, `GOODS_PRICE_CREDIT_RATIO`) and
three missingness flags (`EXT_SOURCE_1/2/3_MISSING`).

**Performance (held-out test set, never seen during tuning):**

| Metric                                    | Value            |
| ----------------------------------------- | ---------------- |
| Test ROC-AUC (point estimate)             | 0.7712           |
| Bootstrap mean ROC-AUC (1000 resamples)   | 0.7713           |
| 95% CI                                    | [0.7650, 0.7777] |
| Full 5-fold CV ROC-AUC (tuned model)      | 0.7666 ± 0.0012  |
| Baseline (untuned) LightGBM — CV / test   | 0.7617 / 0.7658  |
| Baseline Logistic Regression — CV ROC-AUC | 0.7489           |
| Baseline Random Forest — CV ROC-AUC       | 0.7223           |

The tuned model improved on the untuned baseline by roughly half a
percentage point of AUC, consistently across CV and the held-out test set.
This is a real but modest gain — the model is meaningfully better than
random (AUC 0.5) and better than the simpler baselines, but it is not a
dominant, near-perfect classifier. An AUC around 0.77 means there is
substantial, irreducible overlap between defaulting and non-defaulting
applicants in this feature space.

---

## 2. Core Risk: Heavy Dependence on External Bureau Scores

SHAP analysis on the held-out test set shows the top 5 features by mean
|SHAP value| are:

| Rank | Feature                    | Mean \|SHAP\| |
| ---- | -------------------------- | ------------- |
| 1    | `EXT_SOURCE_3`             | 0.3394        |
| 2    | `EXT_SOURCE_2`             | 0.3158        |
| 3    | `CREDIT_TERM_RATIO`        | 0.1416        |
| 4    | `EXT_SOURCE_1`             | 0.1388        |
| 5    | `GOODS_PRICE_CREDIT_RATIO` | 0.1060        |

Three of the top four features by importance are external credit bureau
scores (`EXT_SOURCE_1/2/3`) — not features Home Credit itself controls or
fully understands the construction of. The beeswarm plot confirms the
direction is sane (low external score → higher predicted default risk,
high score → lower risk), which is reassuring from a sanity-check
standpoint, but it does not change the structural risk:

- **If an external bureau's scoring methodology changes**, this model's
  single most influential signal shifts without any change to the
  applicant pool itself. The model would need re-validation, not just
  re-training, since the _meaning_ of the input would have changed.
- **Applicants with thin or no bureau history are systematically
  underserved by this model's strongest signal.** The feature engineering
  step explicitly created `EXT_SOURCE_1/2/3_MISSING` flags because EDA
  Finding 2 established that missingness in these scores itself carries
  predictive signal — meaning the model has _already learned_ to treat
  "no bureau history" as informative. This is statistically useful but
  ethically double-edged: applicants without an external credit history
  (which disproportionately includes younger applicants, recent
  immigrants, or those outside traditional credit systems) may be
  scored less reliably, or scored based on the _absence_ of data rather
  than their actual creditworthiness.

The exact missingness rates and default-rate splits from `02_eda.py`:
`EXT_SOURCE_1` missing 56.4% (default rate: 8.52% missing vs. 7.50%
present); `EXT_SOURCE_2` missing 0.2% (7.88% vs. 8.07% — essentially
no signal from missingness alone); `EXT_SOURCE_3` missing 19.8% (9.31%
vs. 7.77% — the largest present/missing gap of the three). Notably,
`EXT_SOURCE_1` is absent for over half the applicant population, yet
it still ranks 4th in SHAP importance when present — meaning the model
relies on substantially different signals for the 56.4% of applicants
who have no `EXT_SOURCE_1` score at all.

---

## 3. Confirmed Behavioral Finding: The DAYS_EMPLOYED Anomaly Inverts Naive Intuition

EDA Finding 1 (`02_eda.py`) found that applicants with the
`DAYS_EMPLOYED` sentinel anomaly (365243 — confirmed 55,374 rows /
18.0% of the dataset, all of them `NAME_INCOME_TYPE == "Pensioner"`)
default **less** often than the general population: **5.4% vs. an
overall baseline of 8.07%** (the non-anomalous group specifically
defaults at 8.66%, which is higher than the overall rate because the
safer pensioner group pulls the aggregate down).

This is counter to a naive "no current employment = higher risk"
assumption. Finding 4 confirmed — not merely suggested — that this is
a pension-income-stability effect: **100.0%** of anomalous-`DAYS_EMPLOYED`
applicants are `NAME_INCOME_TYPE == "Pensioner"`, an exact identity by
construction in this dataset, not a correlation.

**Risk implication:** if this relationship is driven by pension income
being unusually _stable_ (rather than low), the model may not generalize
well to other forms of non-employment (e.g., unemployment due to job loss
or economic downturn) that this dataset's pensioner-heavy anomaly group
does not represent. A future applicant pool with a different unemployment
composition (e.g., during a recession, with more recently-laid-off
applicants rather than retirees) could see this learned relationship
break down — the model would still predict "anomalous DAYS_EMPLOYED →
lower risk," which may no longer hold.

Finding 4's cross-tab confirmed this conclusively: **100.0%** of
anomalous-`DAYS_EMPLOYED` applicants are `NAME_INCOME_TYPE ==
"Pensioner"` — an exact identity, not a correlation. Among income types
with n ≥ 50, Pensioner has the lowest default rate (5.39%), followed by
State servant (5.76%), Commercial associate (7.48%), and Working
(9.59%, the largest group at n=158,774). The model has therefore learned
a real, stable signal, not a dataset artifact — but it is a signal
specific to pensioners as a population, and the concern raised above
(about non-pensioner unemployment during a recession) remains valid
precisely because that population is almost entirely absent from the
training data (Unemployed: n=22, excluded from headline comparison as
unreliable).

---

## 4. Engineered Ratio Features: Defensible but Unverified Edge Behavior

`CREDIT_TERM_RATIO`, `GOODS_PRICE_CREDIT_RATIO`,
`ANNUITY_INCOME_RATIO`, and `CREDIT_INCOME_RATIO` are all division-based
features. `03_feature_engineering.py` explicitly checks for and replaces
infinite values (from zero denominators) with NaN, which the pipeline
then imputes via `SimpleImputer(strategy="median")`.

This is a sound defensive pattern, but it means **a meaningful subset of
applicants — anyone with `AMT_INCOME_TOTAL`, `AMT_CREDIT`, or
`AMT_ANNUITY` of exactly zero — get a median-imputed value for these
ratios rather than a value reflecting their actual application.** If such
applicants are rare, this is a negligible approximation. If they are not
rare (e.g., a data entry convention where missing financial fields are
recorded as 0 rather than NaN), the model is silently treating a
data-quality issue as an average-risk applicant, which could understate
risk for that group.

**[PENDING]:** the exact count of zero/infinite-ratio rows flagged by
`03`'s defensive check was printed conditionally
(`if n_infinite > 0`) but not shown in this conversation — needed to know
whether this affects 5 rows or 5,000.

---

## 5. Imbalance and Threshold Sensitivity

The dataset's `TARGET` is imbalanced: 91.93% non-default (0), 8.07%
default (1), confirmed from `01_data_prep.py`'s run output.

ROC-AUC is threshold-independent and was the right choice for model
_selection_ and _tuning_ — but it does not by itself tell you what
happens at any specific decision threshold a real loan-approval process
would actually use. This model has not yet been evaluated at a specific
operating threshold with corresponding precision/recall/false-positive
and false-negative _rates_, nor has a cost framework (false negative =
approved loan that defaults; false positive = rejected applicant who
would have repaid) been applied. Per Day 9's threshold-as-business-decision
framing, that analysis — not yet performed for this capstone — would be
necessary before this model could responsibly inform an actual approve/reject
decision, as opposed to a risk _ranking_.

---

## 6. Distributional Drift

This model is trained and evaluated entirely on a single historical
snapshot (`application_train.csv`, 307,511 rows). No temporal
out-of-sample test (e.g., evaluating on applications from a later period
than the training data) has been performed. Macroeconomic shifts
(interest rates, regional employment conditions, inflation affecting
`AMT_INCOME_TOTAL`'s real purchasing power over time) are not represented
in a single static snapshot, and this model carries no explicit mechanism
for detecting when its inputs have drifted from the distribution it was
trained on.

---

## 7. Summary of What Would Need to Happen Before Production Use

1. Quantify the `EXT_SOURCE_*` missingness population size and confirm
   the model's behavior for that group is acceptable, not just
   statistically convenient.
2. Resolve the DAYS_EMPLOYED-anomaly / pensioner overlap question with
   Finding 4's actual numbers, to know whether the learned relationship
   is robust or dataset-specific.
3. Quantify how many rows hit the zero/infinite-ratio edge case in
   feature engineering.
4. Perform threshold-specific cost analysis (false negative vs. false
   positive cost) rather than relying on ROC-AUC alone.
5. Establish a drift-monitoring plan, since this model has only been
   validated on a single historical snapshot.

This report intentionally does not include a "deploy / do not deploy"
verdict — that decision depends on business-side risk tolerance this
technical analysis cannot supply on its own.
