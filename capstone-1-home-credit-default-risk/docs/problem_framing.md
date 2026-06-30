# Problem Framing — Home Credit Default Risk

## Who is the client?

Home Credit, a lender focused on extending credit to people with thin or no
traditional credit history (the "unbanked" or "underbanked"). Their core
business problem: many applicants have no formal credit bureau record, so
standard credit scoring fails them — yet Home Credit still needs to decide
whether lending to them is safe.

## What decision does this model support?

A binary prediction at loan application time: will this applicant default
(TARGET = 1) or repay (TARGET = 0)? The model's output (a probability of
default) feeds into a lending decision — approve, deny, or price the loan
differently (interest rate, required collateral).

## Why this matters (the real stakes)

- **False negative** (predicted safe, actually defaults): direct financial
  loss to Home Credit — the loan amount, minus recovery.
- **False positive** (predicted risky, actually would have repaid): a
  creditworthy person — often from the exact underbanked population Home
  Credit exists to serve — gets denied access to credit. This is the
  population-level harm that makes this dataset ethically interesting, not
  just technically interesting.

This asymmetry means the threshold decision later in the project is a
business/ethics decision, not just a technical one (same lens as the Day 9
threshold work, but with real stakes attached to a real company's mission).

## Dataset

- **Source:** Kaggle competition `home-credit-default-risk`
- **Primary table:** `application_train.csv` — 307,511 rows, 122 columns,
  one row per loan application, identified by `SK_ID_CURR`
- **Target:** `TARGET` (0 = repaid, 1 = defaulted), ~8% positive class
- **Known data quirks to handle explicitly:**
  - `DAYS_EMPLOYED` contains a sentinel value of `365243` for
    pensioners/unemployed — not a real value, must be treated as missing
    (flag with a boolean column before nulling it out)
  - `DAYS_BIRTH`, `DAYS_EMPLOYED`, etc. are negative day-counts relative to
    application date — need sign-flip + conversion to interpretable units
  - Significant missingness across many columns — will need a documented
    imputation strategy, not blanket mean-fill

## Known biases (to expand further once EDA is done)

- The training population reflects Home Credit's existing applicant pool —
  not the general population. Any deployed model inherits whatever
  selection bias exists in who applies to Home Credit in the first place.
- Demographic and regional features are present (e.g. region rating,
  organization type) — these need scrutiny for proxy discrimination even
  if protected attributes aren't used directly. This will be a focus of the
  risk report.

## Primary metric

**ROC-AUC** — matches the original competition's evaluation metric, and is
appropriate given the ~92/8 class imbalance (accuracy would be misleading —
a model predicting "no default" for everyone scores ~92% accuracy while
being useless).

## Success criteria for this capstone

- Beat a simple baseline (majority-class / logistic regression on raw
  features) with a tuned gradient-boosted model
- At least 3 original EDA findings beyond what every public notebook on
  this dataset already covers
- SHAP explanation for top 5 features, framed for a non-technical loan
  officer audience
- An honest risk report — not boilerplate
