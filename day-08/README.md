# Day 08 — Linear Regression: From Math to Code

## What this day covers

Linear Regression implemented three ways, verified to produce identical coefficients,
plus a diagnostic toolkit for the four core OLS assumptions.

---

## Standard task — Three implementations

| Implementation             | Mechanism                                     | When to use                   |
| -------------------------- | --------------------------------------------- | ----------------------------- |
| Normal Equation            | `θ = (XᵀX)⁻¹ Xᵀy` — one-shot matrix inversion | Small datasets, few features  |
| Batch Gradient Descent     | Iterative weight updates via `θ -= lr · ∇MSE` | Large datasets, many features |
| sklearn `LinearRegression` | SVD-based solver (stable Normal Equation)     | Production / baseline         |

**Verification result:** All three produce coefficients within `1e-4` of each other. ✓
**Dataset:** California Housing (`housing.csv`, 20,640 rows — Kaggle)

---

## Out-of-box challenge — Assumption diagnostics

Linear Regression (OLS) requires four assumptions. When they break, the model still
_runs_ — it just produces coefficients you can't trust.

### 1. Linearity

**How it breaks:** True relationship is quadratic; linear model is fit.
**What you see:** A curved arc in the residual-vs-fitted plot — systematic, not random.
**Fix:** Add polynomial/interaction features, or switch to a non-linear model.

### 2. Independence (no autocorrelation)

**How it breaks:** Time-series data with AR(1) errors (ρ = 0.92).
**What you see:** A wave pattern in residuals — runs of positive then negative errors.
**Fix:** Use ARIMA, add lag features, or use GLS (Generalized Least Squares).

### 3. Homoscedasticity (constant variance)

**How it breaks:** Noise variance grows proportionally to X (σ = 0.5·X).
**What you see:** Funnel/trumpet shape — residual spread widens with fitted values.
**Fix:** Log-transform y, use Weighted Least Squares, or apply RobustScaler on target.

### 4. Normality of residuals

**How it breaks:** Errors drawn from a Laplace distribution (heavy tails).
**What you see:** QQ-plot tails diverge from the diagonal in an S-curve.
**Fix:** Use `HuberRegressor` (robust to outliers) or investigate tail-driving observations.

---

## Outputs

| File                           | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `01_three_implementations.png` | Loss curve + predicted vs actual + residual plot  |
| `02_assumption_violations.png` | 4×2 grid: residual plots + QQ-plots per violation |

---

## Key insight

Most DS candidates can _run_ Linear Regression. The diagnostic framework here answers
a harder question: **when is the model lying to you, and why?** Each assumption
violation produces a distinct visual signature — learning to read residual plots and
QQ-plots is a permanent diagnostic skill, not just a Day 8 exercise.

---

## Commit

`day-08: linear regression from scratch + assumption diagnostics`
