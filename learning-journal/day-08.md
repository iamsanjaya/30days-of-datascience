# Day 08 — June 10, 2025

## What I built today

Three independent implementations of Linear Regression (Normal Equation, batch gradient
descent from scratch, sklearn) on the real California Housing dataset (Kaggle, 20,640 rows).
Verified all three produce coefficients within 1e-4 of each other.

## The out-of-box challenge result

Each assumption violation produces a visually distinct pattern:

- Non-linearity → curved arc in residuals
- Autocorrelation → wave / run pattern in residuals
- Heteroscedasticity → funnel shape (spread grows with fitted values)
- Non-normal residuals → S-curve tail divergence in QQ-plot

These are permanent diagnostic signatures. The model runs regardless — it just silently
produces untrustworthy coefficients when assumptions break. The fix for each is different,
which is why diagnosis precedes treatment.

## What surprised me

The gradient descent coefficients converged to within 7.6e-5 of the closed-form solution
after 1000 iterations with lr=0.01 — tighter than expected. Also: the AR(1) autocorrelation
(ρ=0.92) produces a wave pattern that looks almost sinusoidal, even though the generating
process has no sine anywhere.

## What I don't fully understand yet

Why `np.linalg.pinv` (pseudoinverse) is preferred over `np.linalg.inv` for the Normal
Equation — needs deeper understanding of rank-deficiency and what the pseudoinverse does
when XᵀX is singular or near-singular.

## GitHub commit made: ✅

`day-08: linear regression from scratch + assumption diagnostics`

## Tomorrow's priority: Day 09 — Logistic Regression, ROC/AUC, threshold as a business decision
