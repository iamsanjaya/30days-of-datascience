# Learning Journal — Capstone 1: Home Credit Default Risk

Following the Day 24–25 roadmap entry: structured-data ML capstone,
production-grade project format (problem framing → EDA → feature
engineering → model selection → tuning → evaluation → risk report).

---

## 01 — Data Prep

**What I built:** Loaded `application_train.csv`, validated it against
the official dataset's known shape (307,511 × 122, confirmed: 65 float64

- 41 int64 + 16 object columns), confirmed zero duplicate `SK_ID_CURR`
  values, and handled the `DAYS_EMPLOYED` sentinel anomaly — same category
  of issue as the `TotalCharges` whitespace problem from the churn dataset
  earlier in the roadmap: a value that looks numeric but isn't real data,
  and `.isnull()` alone wouldn't have caught it.

**Confirmed numbers:** 55,374 anomalous `DAYS_EMPLOYED` values, exactly
18.0% of all rows. `TARGET` distribution: 91.93% non-default (0), 8.07%
default (1) — confirming this is a genuinely imbalanced classification
problem, consistent with the ~8% default rate commonly cited for this
competition.

**What surprised me:** Nothing at this stage — the anomaly rate (18.0%)
and class imbalance (8.07%) both landed close to what's commonly
reported for this dataset, so this step confirmed expectations rather
than overturning them. The real surprises came in `02`.

**What I don't fully understand yet:** Why Home Credit chose the
specific sentinel value `365243` for "not applicable" rather than a more
obviously-fake number. It's suspiciously close to 1000 years in days
(365243 / 365.25 ≈ 999.99) but not exact — possibly a leap-year-aware
date library artifact from whatever internal system computed it. Not
something that blocks any downstream work, but it's the kind of "where
did this number actually come from" question that's stayed unresolved.

**GitHub commit made:** ✅

---

## 02 — EDA

**What I built:** Four targeted findings rather than an exhaustive
column-by-column dump — deliberately chosen to go past what every public
Home Credit notebook already covers.

**Finding 1 — `DAYS_EMPLOYED` anomaly vs. default:** Confirmed and
sharp. Non-anomalous applicants (n=252,137) default at 8.66%; anomalous
applicants (n=55,374) default at 5.40% — anomalous applicants are
meaningfully _safer_, counter to a naive "no current employment = higher
risk" assumption. Both groups are large enough that this isn't a
small-sample fluke.

**Finding 2 — `EXT_SOURCE_*` missingness as signal:** Confirmed, but
the magnitude and scale differ dramatically across the three columns:

- `EXT_SOURCE_1`: missing **56.4%** of the time — over half the dataset
  has no score at all. Default rate if missing: 8.52% vs. 7.50% if
  present. Missingness nudges risk up, but only mildly given how
  dominant this column is in SHAP (later finding).
- `EXT_SOURCE_2`: missing only **0.2%** of the time — nearly always
  present, so the missingness flag for this column captures almost
  nobody and carries almost no predictive signal despite `EXT_SOURCE_2`
  being the second-most important feature overall.
- `EXT_SOURCE_3`: missing **19.8%** of the time — default rate if
  missing: 9.31% vs. 7.77% if present, the largest present/missing gap
  of the three.

**Finding 3 — credit/income ratio deciles:** Not monotonic. Default
rate starts at 6.91% in the lowest decile, climbs to a peak of 9.20% in
the 6th decile (ratio range 3.265–3.906), then declines back to 7.07%
in the highest decile (range 7.488–84.737). Risk peaks at _moderate_
leverage, not extreme leverage — the chart title the code generated was
"Default Risk Peaks at Moderate Leverage, Not Extreme Leverage."

**Finding 4 — is the anomaly really about pensioners?** Completely
confirmed, not just suggestive: **100.0%** of anomalous-`DAYS_EMPLOYED`
applicants are `NAME_INCOME_TYPE == "Pensioner"`. This isn't a
correlation — it's an exact identity by construction in this dataset.
Income-type default rates (for groups with n ≥ 50):

- Pensioner: 5.39% (lowest risk, matching Finding 1 exactly since they
  are the same population)
- State servant: 5.76%
- Commercial associate: 7.48%
- Working: 9.59% (highest risk among reliable groups, n=158,774)
- Unemployed (n=22, 36.4%) and Maternity leave (n=5, 40.0%) were
  correctly excluded from the headline comparison as unreliable.

**What surprised me:** Two things. First, Finding 4 being a total
overlap rather than partial — I'd expected "mostly pensioners" with some
other anomaly causes mixed in, not literally 100%. Second, Finding 3's
peak-in-the-middle shape — I'd have guessed default risk rises steadily
with leverage, not peaks and falls. The highest-leverage applicants are
actually _less_ risky than the middle-leverage group, which doesn't fit
the simple mental model.

**What I don't fully understand yet:** Why default risk _declines_ in
the highest credit/income decile (Finding 3). A plausible explanation is
that very-high-ratio applicants are self-selected — they only get
approved at all when they have strong compensating factors (high
`EXT_SOURCE_*` scores, stable income type like State servant or
Commercial associate) that this single-feature decile view can't see.
But I haven't verified whether high-decile applicants actually have
higher `EXT_SOURCE_*` scores or safer income types — this is a
hypothesis from reasoning about selection effects, not something
confirmed in the data.

**GitHub commit made:** ✅

---

## 03 — Feature Engineering

**What I built:** Row-wise engineered features with no fitted
statistics — age/tenure year-conversions (`AGE_YEARS`, `YEARS_EMPLOYED`,
`YEARS_REGISTRATION`, `YEARS_ID_PUBLISH`), four financial ratios
(`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM_RATIO`,
`GOODS_PRICE_CREDIT_RATIO`), and `EXT_SOURCE_1/2/3_MISSING` boolean
flags promoting Finding 2's missingness insight directly into a usable
feature. Deliberately deferred imputation, encoding, and scaling to the
Pipeline objects in `04`/`05` — anything requiring a fitted statistic
has to happen post-split to avoid leaking test-set information into
preprocessing, even without touching `TARGET` directly.

**What surprised me:** How directly each engineered feature traces back
to a specific EDA finding rather than being added speculatively. The
`EXT_SOURCE_*_MISSING` flags in particular feel like the cleanest
example: Finding 2 established that missingness itself predicts default,
so the flags go straight into the feature set — not as an imputation
detail, but as a first-class feature. That felt like the "feature
engineering wins competitions" principle from the roadmap playing out
concretely.

**What I don't fully understand yet:** Given that `EXT_SOURCE_2` is
missing only 0.2% of the time, its `_MISSING` flag is nearly always
False and covers only ~617 applicants — it's hard to see how that flag
can contribute anything meaningful to the model, yet it still gets
passed through as a feature. In retrospect, it might have been worth
either dropping it or noting explicitly in the code that it's included
for symmetry/consistency rather than expected signal. Whether including
a near-constant feature actually _hurts_ the model (by adding noise to
the feature importance signal) vs. is simply neutral, I haven't tested.

**GitHub commit made:** ✅

---

## 04 — Baseline Model Comparison

**What I built:** Logistic Regression, Random Forest, and LightGBM,
each in an identical-preprocessing sklearn Pipeline, compared via
Stratified 5-fold CV (ROC-AUC) and a single held-out test evaluation.

**Full confirmed results:**

| Model               | CV ROC-AUC | CV Std  | Test ROC-AUC |
| ------------------- | ---------- | ------- | ------------ |
| Logistic Regression | 0.7489     | ±0.0024 | 0.7522       |
| Random Forest       | 0.7223     | ±0.0035 | 0.7246       |
| LightGBM            | 0.7617     | ±0.0015 | 0.7658       |

LightGBM won on both CV and test by a clear margin. Selected for Optuna
tuning in `05`.

**What surprised me:** Two things, both about Logistic Regression. First,
it outperformed Random Forest by nearly 2.7 percentage points despite
being the simpler model — on a dataset this wide (262 features after
one-hot encoding, many sparse categorical columns), the linear model
apparently generalizes better than the tree ensemble with default
hyperparameters. Second and more striking: Logistic Regression's test
AUC (0.7522) came out _higher_ than its CV mean (0.7489), even though
`ConvergenceWarning` fired on every single CV fold — lbfgs genuinely
didn't converge in 1000 iterations in any fold. An unconverged model
that still generalizes better-than-expected to the test set is a result
I don't fully have a clean explanation for.

Also worth noting: Random Forest triggered a `joblib` worker crash
warning ("a worker stopped while some jobs were given to the executor")
mid-CV run at around the 766s mark. The CV result still came through
(0.7223), but that warning signals the Kaggle session was under memory
pressure during the RF run — the 200-estimator RF on 246,008 training
rows with 262 features is genuinely expensive, and this is probably why
RF's `cv_std` (±0.0035) was the highest of the three models.

**What I don't fully understand yet:** Why Logistic Regression didn't
converge (lbfgs hitting `max_iter=1000`) but still produced a
reasonable and arguably generalizing result. The `ConvergenceWarning`
means the optimization found a direction of improvement it couldn't
finish following — not that the solution is wrong, but that it's not
the true optimum. I'd expect an unconverged model to underperform, not
match or exceed a converged one. One explanation is that early stopping
due to iteration limit acts as a form of implicit regularization on this
dataset — preventing the model from over-optimizing the training set
within each fold. But that's speculative; I haven't tested what happens
at `max_iter=5000` or with a different solver like `saga`, which is
designed for large sparse datasets and might actually converge here.

**GitHub commit made:** ✅

---

## 05 — Optuna Tuning

**What I built:** Tuned LightGBM via Optuna's TPE sampler (50 trials,
3-fold CV search, full 5-fold CV re-evaluation for the reported number),
with every trial logged to MLflow (SQLite backend) and the Optuna study
using persistent SQLite storage so a killed Kaggle session resumes from
the last completed trial rather than restarting from zero.

**Result:** Best trial (3-fold search): 0.7656. Full 5-fold CV on
winning params: 0.7666 ± 0.0012. Held-out test: 0.7712. Tuning gave a
real, consistent ~0.5 percentage-point improvement over the untuned
baseline's 0.7617 CV / 0.7658 test, and the CV → test relationship
stayed healthy throughout — no sign of tuning overfitting to the CV
folds.

**What surprised me:** Saving the joblib pipeline _before_ the MLflow
model-logging step mattered in practice, not just in theory — MLflow's
skops-based serializer initially balked at LightGBM's internal Booster
class as an "untrusted type." Wrapping that logging call in its own
`try/except` meant the actual deliverable (the saved model) was never
at risk from a tracking-tool's security feature. This is the kind of
failure mode that's easy to reason about in the abstract but genuinely
only feels important once you've seen it nearly cause a problem.

**What I don't fully understand yet:** Optuna converged on a very low
`learning_rate` (0.0199) combined with a high `n_estimators` (890) —
slow learning with many rounds. This makes intuitive sense as a
bias-variance trade-off: smaller steps are more precise but need more
of them. But I haven't verified whether this is genuinely optimal or
whether Optuna's TPE sampler happened to explore this region of the
parameter space early and kept exploiting it, possibly missing a
higher-learning-rate / fewer-estimator combination that would score
similarly but train faster. The Day 16 challenge was specifically about
this exploration-exploitation tension in Bayesian optimization, and I
haven't resolved it empirically for this model.

**GitHub commit made:** ✅

---

## 06 — Final Evaluation + SHAP

**What I built:** A fast, read-only evaluation script — no retraining,
no Optuna. Loads `lightgbm_tuned.joblib`, reproduces the identical
train/test split from `04`/`05` (same params on the same source data →
same split, deterministically), then bootstraps 1000 resamples of the
already-computed test predictions for a 95% CI on ROC-AUC, and runs
SHAP's `TreeExplainer` on the raw `LGBMClassifier` fed through the
fitted `ColumnTransformer`, with real post-encoding feature names.

**Result:** Point-estimate test ROC-AUC 0.7712 matched `05`'s reported
number exactly — confirming the split reproduction is genuinely correct,
not a coincidence. Bootstrap CI: 0.7713 mean, [0.7650, 0.7777] 95%
interval (tight, well-centered). SHAP top-5:

| Rank | Feature                    | Mean \|SHAP\| |
| ---- | -------------------------- | ------------- |
| 1    | `EXT_SOURCE_3`             | 0.3394        |
| 2    | `EXT_SOURCE_2`             | 0.3158        |
| 3    | `CREDIT_TERM_RATIO`        | 0.1416        |
| 4    | `EXT_SOURCE_1`             | 0.1388        |
| 5    | `GOODS_PRICE_CREDIT_RATIO` | 0.1060        |

Three external bureau scores dominate — confirmed by the beeswarm plot
showing a clean directional pattern: low score (blue) → positive SHAP
(higher predicted risk), high score (red) → negative SHAP (lower risk).
Two of the engineered ratio features earned real places in the top 5.

**What surprised me:** How cleanly the SHAP beeswarm plot's
directionality came out — no inverted or noisy-looking bands. That's a
genuine sanity check I didn't take for granted. More notable: finding
`EXT_SOURCE_2` as the second most important feature by SHAP, despite it
being missing only 0.2% of the time (Finding 2). The `EXT_SOURCE_2`
_value_ is extremely influential when present — it's almost always
available, and when it is, it's carrying strong signal. That's a
different story from `EXT_SOURCE_1`, which is missing 56.4% of the time
and still ranking 4th — meaning the model is heavily relying on a column
that's absent for over half the applicant population.

**A real bug, not just a typing nitpick:** Pylance flagged `fig_summary`
as undefined, and on inspection it was a genuine latent bug —
`shap.summary_plot()` manages its own matplotlib figure internally
rather than drawing onto a figure handle created beforehand, so the fix
was capturing the figure via `plt.gcf()` _after_ the plot call, not
before. Worth remembering as a SHAP-specific gotcha distinct from
standard matplotlib usage.

**What I don't fully understand yet:** `EXT_SOURCE_1` is missing for
56.4% of all applicants but still ranks 4th in SHAP importance — which
means the model is making confident predictions for over half the
applicant pool using a different set of signals than the remaining half.
For the 56.4% with no `EXT_SOURCE_1`, the model falls back on
`EXT_SOURCE_2`/`3` (if present), the ratio features, and the imputed
median — but I haven't checked whether the model's calibration (not just
its discrimination/AUC) holds equally well for the missing vs. present
subgroup. AUC is an aggregate metric and can look identical for two
groups even when one is badly miscalibrated. That subgroup analysis
hasn't been done.

**GitHub commit made:** ✅

---

## Capstone-Level Reflection

**What changed across this capstone:** Going from `01` through `06`
made the leakage-avoidance discipline from Day 15's pipeline lesson feel
concrete rather than abstract. Every script either computes something
purely row-wise (safe pre-split: `03`) or explicitly defers fitted
statistics into a Pipeline object fit only on training folds (`04`, `05`).
That distinction showed up as real code decisions multiple times — the
`EXT_SOURCE_*_MISSING` flags computed before the split in `03`, while
the actual _imputation_ of those same columns was deferred to `04`/`05`'s
`SimpleImputer` inside the Pipeline.

**What this capstone confirmed that was previously just a principle:**
Feature engineering from EDA findings rather than speculation produces
features that actually matter — two of the top 5 SHAP features
(`CREDIT_TERM_RATIO`, `GOODS_PRICE_CREDIT_RATIO`) are engineered
features that didn't exist in the raw data and traced directly back to
EDA observations. That's a different experience than adding features
because they "seem useful."

**Biggest unresolved question from this capstone:** Whether this model's
test AUC of 0.7712 holds equally well across all applicant subgroups, or
whether it's driven by confident predictions for the 43.6% of applicants
who have all three `EXT_SOURCE_*` scores, masking worse calibration for
the rest. The aggregate AUC doesn't answer this. A proper fairness and
subgroup-calibration analysis was not performed as part of this capstone.

**Next up:** Capstone 2 (Days 26–27) — Deep Learning / NLP project with
live Gradio/Streamlit demo deployed on HuggingFace Spaces.

**GitHub commit made:** ✅
