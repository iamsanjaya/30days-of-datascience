"""
Out-of Box Challenge
Linear Regression has 4 core assumptions. This script:
    - Deliberately violates each one using synthetic data
    - Shows what the residual plot looks like when each assumption breaks
    - Documents in one sentence what to do when each breaks

Assumptions:
    1. Linearity - relationship between X and y is linear
    2. Independence - residuals are not correlated (no autocorrelation)
    3. Homoscedasticity - residuals variance is constant across fitted values
    4. Normality - residuals are normally distributed
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

from config import OUTPUTS_DIR, RANDOM_SEED, DPI

rng = np.random.default_rng(RANDOM_SEED)
N = 300
y1_pred = y2_pred = y3_pred = y4_pred = np.zeros(N)
res1 = res2 = res3 = res4 = np.zeros(N)

# Helper: fit and extract residuals


def fit_and_residuals(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    return y_pred, y - y_pred


# 1. Violation: Non - linearity
# True relationship is quadratic; we fit a linear model.
# Residual plot: curved pattern (systematic arc) - not random scatter.

X1 = rng.uniform(-3, 3, N)
y1 = X1**2 + rng.normal(0, 0.5, N)  # quadratic truth
y1_pred, res1 = fit_and_residuals(X1, y1)


# 2. Violation: Autocorrelation (non-independence)
# Residuals from a time-series where each error depends on the previous one.
# Residuals plot: structured wave pattern - sequential runs above/below zero.

X2 = np.linspace(0, 10, N)
errors: np.ndarray = np.zeros(N)
errors[0] = float(rng.normal(0, 1))
for t in range(1, N):
    errors[t] = 0.92 * errors[t - 1] + float(rng.normal(0, 0.4))  # AR(1) with ρ=0.92
    y2 = 2 * X2 + errors
    y2_pred, res2 = fit_and_residuals(X2, y2)


# 3. Violation: Heteroscedasticity
# Variance of errors grows with X (funnel shape).
# Residual plot: fan/trumpet - spread widens with fitted values.

X3 = rng.uniform(1, 10, N)
noise_scale = 0.5 * X3  # variance = x
y3 = 3 * X3 + rng.normal(0, noise_scale)
y3_pred, res3 = fit_and_residuals(X3, y3)


# 4. Violation: Non-normal residuals
# Errors drawn from a heavy-tailed distribution (Laplace).
# QQ-plot: S-curve (tails diverge from the diagonal).

X4 = rng.uniform(0, 10, N)
y4 = 2 * X4 + rng.laplace(0, 2, N)  # Laplace = heavy tails
y4_pred, res4 = fit_and_residuals(X4, y4)

# Diagnostic plot: 4 rows x 2 cols
# Left column: Residuals vs Fitted
# Right column: QQ-plot of residuals

fig, axes = plt.subplots(4, 2, figsize=(13, 18))
fig.suptitle(
    "Day 08 - Linear Regression Assumption Violations\n"
    "Each row deliberately breaks one assumption",
    fontsize=13,
    fontweight="bold",
)

TITLES = [
    ("1. Non-linearity", "Curved arc in residuals → model missed the quadratic trend"),
    ("2. Autocorrelation", "Wave pattern → residuals depend on their neighbours"),
    ("3. Heteroscedasticity", "Funnel shape → variance grows with fitted value"),
    ("4. Non- normal residuals", "Heavy Laplace errors → QQ-plot tails diverge"),
]
FIXES = [
    "Fix: add polynomial/interaction features or use a non-linear model",
    "Fix: use time-series models (ARIMA) or add lag features",
    "FIX: use WLS, log-transform y, or RobustScaler on target",
    "FIx: use robust regression or check for outliers driving the tail",
]

data = [
    (y1_pred, res1),
    (y2_pred, res2),
    (y3_pred, res3),
    (y4_pred, res4),
]

for row, ((fitted, resid), (title, subtitle), fix) in enumerate(
    zip(data, TITLES, FIXES)
):
    # Left: Residuals vs Fitted

    ax_left = axes[row, 0]
    ax_left.scatter(fitted, resid, alpha=0.35, s=12, color="steelblue")
    ax_left.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax_left.set_title(f"{title}\n{subtitle}", fontsize=9.5, fontweight="bold")
    ax_left.set_xlabel("Fitted values")
    ax_left.set_ylabel("Residuals")
    ax_left.text(
        0.02,
        0.03,
        fix,
        transform=ax_left.transAxes,
        fontsize=8,
        color="darkred",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff0f0", "alpha": 0.8},
    )

    # Right: QQ-plot

    ax_right = axes[row, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax_right.scatter(osm, osr, alpha=0.35, s=12, color="steelblue")
    ax_right.plot(
        [min(osm), max(osm)],
        [slope * min(osm) + intercept, slope * max(osm) + intercept],
        color="red",
        linewidth=1.5,
        linestyle="--",
    )
    ax_right.set_title("QQ-plot of residuals", fontsize=9.5)
    ax_right.set_xlabel("Theoretical quantiles")
    ax_right.set_ylabel("Sample quantiles")

plt.tight_layout()
out_path = OUTPUTS_DIR / "02_assumption_violations.png"
plt.savefig(out_path, dpi=DPI)
plt.close()
print(f"Saved: {out_path}")


# Console summary

print("\n" + "=" * 60)
print("ASSSUMPTION VIOLATION SUMMARY")
print("=" * 60)

diagnostics = [
    (
        "1. Linearity",
        "Residual plot shows a curved arc - the model under-predicts\n"
        " in the middle and over-predicts at the extremes.",
        "Add polynomial features or use a non-linear model.",
    ),
    (
        "2. Independence (autocorrelation)",
        " Residuals form a wave pattern → each error is correlated/n"
        " with the previous one (Durbin-Watson << 2).",
        "Use ARIMA / lag features / GLS for time-series data.",
    ),
    (
        "3. Homoscedasticity",
        "Residuals fan out as fitted values increase - a classic\n"
        " funnel shape indicating variance grows with the mean.",
        "Log-transform y, use WLS, or apply RobustScaler on target.",
    ),
    (
        "4. Normality of residuals",
        "QQ-plot tails deviate sharply from the diagonal - heavy\n"
        " Laplace tails inflate standard errors of coefficients.",
        "Use robust regression (HuberRegressor) or remove outliers.",
    ),
]

for name, observation, fix in diagnostics:
    print(f"\n{name}")
    print(f"Observation: {observation}")
    print(f"Fix: {fix}")

# %%
