# %%
"""
# Day - 04 | Standard Task - Batch Gradient Descent from Scratch
# NumPy only. No sklearn. No scipy.
# Goal : Implement GD for linear regression, experiment with 3 learning rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Reproducibility

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# %%
"""
# 1. Generate Synthetic Data
#     True relationship: y = 3x +7 + gaussian_noise
#     We know the ansswer - so we can verify our GD.
"""
N = 200  # number of data points
TRUE_W = 3.0  # true weight (slope)
TRUE_B = 7.0  # true bias (intercept)

X = np.linspace(-5, 5, N)
# NOTE : X is zero - centered (mean = 0 ). This is intentional - it makes the
# w and b gradients nearly independent, so both parameters coverage at similar
# rates. If X were not centered (e.g. linspace (0,20)), b would lag far behind.
# This is one reason input normalization matter in practice.
noise = np.random.randn(N) * 0.5  # small gaussina noise, std = 0.5
y = TRUE_W * X + TRUE_B + noise  # noise linear signal

print(f"Data generated: {N} samples")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")
print(f"X range: [{X.min():.1f}, {X.max():.1f}]")
print(f"y range: [{y.min():.1f}, {y.max():.1f}]")


# %%
# 2. Loss Function - MEan Squared Error
#    MSE = (1/n) * Σ (ŷᵢ - yᵢ)²


def compute_loss(X, y, w, b):
    """Mean Squared Error between predictions and targets."""
    y_pred = w * X + b
    residuals = y_pred - y
    mse = np.mean(residuals**2)
    return mse


# %%
# 3. Gradient Computation
# ∂L/∂w = (2/n) * Σ (ŷᵢ - yᵢ) * X₁
# ∂L/∂b = (2/n) * Σ (ŷᵢ - yᵢ)

# Why these formulas?
# Take derivative of MSE w.r.t. w using chain rule:
# d/dw [(wX₁ + b - y₁ )² ] = 2(wX₁ + b - y₁ ) * X₁
# Average over n samples ⟶  (2/n) * Σ (ŷᵢ - yᵢ) * X₁


def compute_gradients(X, y, w, b):
    """Compute gradients of MSE w.r.t. w and b."""
    n = len(X)
    y_pred = w * X + b
    residuals = y_pred - y

    grad_w = (2 / n) * np.dot(residuals, X)  # scalar
    grad_b = (2 / n) * np.mean(residuals)  # scalar

    return grad_w, grad_b


# %%
# 4. Batch Gradient Descent
# Repeat for EPOCHS iterations
#   1. Compute loss
#   2. Compute gradients
#   3. Update parameters: w ← w - α * grad_w, b ← b - α * grad_b


def batch_gradient_descent(X, y, learning_rate, epochs=500, w_init=0.0, b_init=0.0):
    """
    Batch gradient descent for linear regression.

    Returns:
    w_final, b_final: learned parameters
    loss_history: list of MSE at each epoch
    """
    w = w_init
    b = b_init
    loss_history = []

    for epoch in range(epochs):
        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)

        # Detect divergence early - NaN/Inf means learning rate too large
        if np.isnan(loss) or np.isinf(loss):
            print(f"Divergence detected at epoch {epoch} - loss= {loss}")
            # Pad remaining epochs with the last valid loss for clean potting
            loss_history.extend([loss_history[-2]] * (epochs - epoch - 1))
            break

        grad_w, grad_b = compute_gradients(X, y, w, b)

        # The core update rule
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

    return w, b, loss_history


# %%
# 5. Experiment with 3 Learning Rates


EPOCHS = 10000

experiments = {
    "Too Small (α = 0.0001)": 0.0001,
    "Just Right (α = 0.05)": 0.05,
    "Too Large (α = 0.5)": 0.5,
}


results = {}

name: str = ""
w_final: float = 0.0
b_final: float = 0.0
loss_hist: list[float] = []

print("\n" + "=" * 60)
print("Gradient Descent Experiments")
print("=" * 60)

for name, lr in experiments.items():
    w_final, b_final, loss_hist = batch_gradient_descent(
        X, y, learning_rate=lr, epochs=EPOCHS
    )
    results[name] = {
        "lr": lr,
        "w": w_final,
        "b": b_final,
        "loss_history": loss_hist,
        "final_loss": loss_hist[-1],
    }


# Cap display for diverged cases

display_loss = loss_hist[-1] if not np.isnan(loss_hist[-1]) else float("inf")
print(f"\n  {name}")
print(f"Learned  w={w_final:.4f}  b={b_final:.4f}")
print(f"True  w={TRUE_W:.4f}  b={TRUE_B:.4f}")
print(f"Final MSE: {display_loss:.4f}")

# %%
# 6. Plot 1 — Loss Curves for All 3 Learning Rates


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Loss Curves — Effect of Learning Rate on Gradient Descent",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)

colors = ["#e74c3c", "#2ecc71", "#e67e22"]

for ax, (name, res), color in zip(axes, results.items(), colors):
    loss_hist = res["loss_history"]

    # Clip extreme values so diverged plot is still readable

    clipped = np.clip(loss_hist, 0, loss_hist[0] * 3)

    ax.plot(clipped, color=color, linewidth=2)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    # Annotate final loss on the plot
    if not np.isnan(res["final_loss"]) and not np.isinf(res["final_loss"]):
        ax.annotate(
            f"Final MSE: {res['final_loss']:.2f}",
            xy=(EPOCHS * 0.6, clipped[-1]),
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color),
        )

plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {loss_plot_path}")


# %%
# 7. Plot 2 — Predicted Line vs Actual Data (for "Just Right" LR)


best = results["Just Right (α = 0.05)"]
w_pred = best["w"]
b_pred = best["b"]

x_line = np.linspace(X.min(), X.max(), 200)
y_true_line = TRUE_W * x_line + TRUE_B  # the true underlying line
y_pred_line = w_pred * x_line + b_pred  # what GD learned

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(X, y, alpha=0.35, s=20, color="#95a5a6", label="Data (noisy)")
ax.plot(
    x_line,
    y_true_line,
    color="#2c3e50",
    linewidth=2.5,
    linestyle="--",
    label=f"True line  (w={TRUE_W}, b={TRUE_B})",
)
ax.plot(
    x_line,
    y_pred_line,
    color="#2ecc71",
    linewidth=2.5,
    label=f"GD learned (w={w_pred:.3f}, b={b_pred:.3f})",
)

ax.set_title(
    "Gradient Descent — Predicted vs True Line", fontsize=13, fontweight="bold"
)
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fit_plot_path = os.path.join(OUTPUT_DIR, "predicted_vs_true.png")
plt.savefig(fit_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {fit_plot_path}")


# %%
# 8. Plot 3 — Residual Plot (how far off are our predictions?)
#    Residuals should be randomly scattered around 0 for a good fit.
#    A pattern in residuals = model is missing something systematic.


y_pred_all = w_pred * X + b_pred
residuals = y - y_pred_all

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residuals vs X

axes[0].scatter(X, residuals, alpha=0.4, s=20, color="#3498db")
axes[0].axhline(0, color="#e74c3c", linewidth=1.5, linestyle="--")
axes[0].set_title("Residuals vs X\n(random scatter = good fit)", fontsize=11)
axes[0].set_xlabel("X")
axes[0].set_ylabel("Residual (y - ŷ)")
axes[0].grid(True, alpha=0.3)

# Residual histogram

axes[1].hist(residuals, bins=25, color="#3498db", edgecolor="white", alpha=0.8)
axes[1].set_title("Residual Distribution\n(should be roughly normal)", fontsize=11)
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Count")
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    "Residual Diagnostics — Gradient Descent Linear Regression",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
residual_plot_path = os.path.join(OUTPUT_DIR, "residuals.png")
plt.savefig(residual_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {residual_plot_path}")


# %%
# 9. Summary Report

print("\n" + "=" * 60)
print("SUMMARY — What Did We Learn?")
print("=" * 60)
print(f"\nTrue parameters:    w = {TRUE_W},  b = {TRUE_B}")
print("\nRecovered by GD (lr=0.01):")
print(f"w = {w_pred:.4f}  (error: {abs(w_pred - TRUE_W):.4f})")
print(f"b = {b_pred:.4f}  (error: {abs(b_pred - TRUE_B):.4f})")

print("""
Key observations:
• α =0.0005: Loss decreases, but very slowly. After 500 epochs, still
    far from converged. In practice you'd need ~10x more epochs.

• α =0.01:   Loss drops sharply in early epochs, then flattens near
    the minimum. Recovered parameters are very close to truth.

• α =0.5:    Loss diverges immediately — parameters fly off to infinity.
    The gradient step overshoots the valley bottom each time.
""")
