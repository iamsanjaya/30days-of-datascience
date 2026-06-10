"""
Three implementations of Linear Regression on California Housing data:
1. Normal Equation (closed-form)
2. Batch Gradient Descent (NumPy only)
3. sklearn LinearRegression
All three must produce matching coefficients.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    OUTPUTS_DIR,
    RANDOM_SEED,
    N_ITERATIONS,
    LEARNING_RATE,
    DPI,
)

np.random.seed(RANDOM_SEED)

# 1. Load & prepare data

df = pd.read_csv("/Users/sanjayachaudhary/Desktop/projects/day-08/data/housing.csv")
df = df.dropna()
X_raw = df[["median_income", "total_rooms"]].to_numpy()
y = df["median_house_value"].to_numpy()

# Scale features - critical for gradient descent convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Add bias column (column of ones) for matrix formulations
X_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED
)
X_train_bias = np.hstack([np.ones((X_train_raw.shape[0], 1)), X_train_raw])
X_test_bias = np.hstack([np.ones((X_test_raw.shape[0], 1)), X_test_raw])


# 2. Implementation 1: Normal Equation


def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    θ = (XᵀX)⁻¹ Xᵀy
    Solves for optimal weights analytically.
    X must include a bias column of ones.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y  # pinv for numerical stability


theta_normal = normal_equation(X_train_bias, y_train)
y_pred_normal = X_test_bias @ theta_normal
mse_normal = mean_squared_error(y_test, y_pred_normal)
r2_normal = r2_score(y_test, y_pred_normal)

print("=" * 55)
print("1. Normal Equation")
print(f"Coefficients: {theta_normal}")
print(f"MSE: {mse_normal:.4f}")
print(f"R²: {r2_normal:.4f} ")


# 3. Implementation 2: Batch Gradient Descent


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = LEARNING_RATE,
    n_iter: int = N_ITERATIONS,
) -> tuple[np.ndarray, list[float]]:
    """
    Minimize MSE using batch gradient descent.
    Update rule: θ = θ - (lr/m) * Xᵀ(Xθ - y)
    X must include a bias column of ones.
    """
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    loss_history: list[float] = []

    for _ in range(n_iter):
        residuals = X @ theta - y
        gradient = (1 / m) * X.T @ residuals
        theta -= lr * gradient
        loss_history.append(float(np.mean(residuals**2)))
    return theta, loss_history


theta_gd, loss_history = batch_gradient_descent(X_train_bias, y_train)
y_pred_gd = X_test_bias @ theta_gd
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

print("\n2. Batch Gradient Descent")
print(f"Coefficients: {theta_gd}")
print(f"MSE: {mse_gd: .4f}")
print(f"R²: {r2_gd: .4f}")


# 4. Implementation 3: sklearn LinearRegression

sk_model = LinearRegression()
sk_model.fit(X_train_raw, y_train)
y_pred_sk = sk_model.predict(X_test_raw)
mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)


# sklearn separated intercept: assemble for comparison

theta_sk = np.concatenate([np.array([sk_model.intercept_]), sk_model.coef_])

print("\n3. sklearn LinearRegression")
print(f"Coefficients: {theta_sk}")
print(f"MSE: {mse_sk:.4f}")
print(f"R²: {r2_sk:.4f}")

# 5. Coefficient verification
print("\n" + "=" * 55)
print("Coefficient Comparison (must be near-identical):")
print(f"Normal Eq: {theta_normal}")
print(f"Grad Desc: {theta_gd}")
print(f"sklearn: {theta_sk}")

max_diff_ne_sk = float(np.max(np.abs(theta_normal - theta_sk)))
max_diff_gd_sk = float(np.max(np.abs(theta_gd - theta_sk)))
print(f"\n Max |Normal Eq - sklearn|: {max_diff_ne_sk:.6f}")
print(f"Max |Grad Desc - sklearn|: {max_diff_gd_sk:.6f}")

if max_diff_ne_sk < 1e-4 and max_diff_gd_sk < 1e-4:
    print("All three implementation agree.")
else:
    print("Mismatch detected - check scaling or iterations.")

# 6. Plots

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Day 08 - Linear Regression: Three Implementation", fontsize=13, fontweight="bold"
)

# Loss curve (gradient descent)

axes[0].plot(loss_history, color="steelblue", linewidth=1.5)
axes[0].set_title("Gradient Descent Loss Curve")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MSE")
axes[0].set_yscale("log")

# Prediction vs Actual (sklearn)

axes[1].scatter(y_test, y_pred_sk, alpha=0.3, s=10, color="steelblue")
lims = [min(y_test.min(), y_pred_sk.min()), max(y_test.max(), y_pred_sk.max())]
axes[1].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
axes[1].set_title(f"Predicted vs Actual (R² = {r2_sk:.3f})")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")
axes[1].legend()


# Residual plot (sklearn)

residuals_sk = y_test - y_pred_sk
axes[2].scatter(y_pred_sk, residuals_sk, alpha=0.3, s=10, color="steelblue")
axes[2].axhline(0, color="red", linewidth=1.5, linestyle="--")
axes[2].set_title("Residuals vs Fitted (sklearn)")
axes[2].set_xlabel("Fitted values")
axes[2].set_ylabel("Residuals")


plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "01_three_implementations.png", dpi=DPI)
plt.close()
print(f"\nSaved: {OUTPUTS_DIR / '01_three_implementations.png'}")
