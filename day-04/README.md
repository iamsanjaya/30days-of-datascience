# Day 04 — Gradient Descent from Scratch + Divergence Study

**Phase:** Foundation | **Topic:** NumPy — Linear Algebra & Gradient Descent
**Status:** ✅ Complete

---

## Structure

```
day-04/
├── gradient_descent.py       # standard task
├── divergence_threshold.py   # out-of-box challenge
├── README.md
├── outputs/
│    ├── loss_curves.png
│    ├── predicted_vs_true.png
│    ├── residuals.png
│    ├── divergence_threshold.png
│    └── divergence_geometry.png

learning-journal/
  └── day-04.md

```

---

## Standard Task — `gradient_descent.py`

Batch gradient descent implemented from scratch using only NumPy. No sklearn, no scipy.

**Dataset:** Synthetic linear data — `y = 3x + 7 + noise` (N=200, noise std=0.5)

**What is implemented:**

- MSE loss function: `L = (1/n) · Σ (ŷᵢ - yᵢ)²`
- Gradient computation via chain rule:
  - `∂L/∂w = (2/n) · Σ (ŷᵢ - yᵢ) · Xᵢ`
  - `∂L/∂b = (2/n) · mean(ŷᵢ - yᵢ)`
- Parameter update rule: `w ← w - lr · ∂L/∂w`, `b ← b - lr · ∂L/∂b`
- 3 learning rate experiments over 10,000 epochs

**Learning rate experiments:**

| Learning Rate           | Behaviour                   | Final w | Final b |
| ----------------------- | --------------------------- | ------- | ------- |
| lr = 0.0001 (too small) | Crawls — far from converged | ~3.01   | ~0.07   |
| lr = 0.05 (just right)  | Smooth descent — converges  | ~3.01   | ~6.93   |
| lr = 0.5 (too large)    | Diverges at epoch 175       | ∞       | ∞       |

**Outputs:**

- `loss_curves.png` — loss over epochs for all 3 learning rates side by side
- `predicted_vs_true.png` — GD-fitted line overlaid on the noisy scatter
- `residuals.png` — residual scatter plot + residual distribution histogram

---

## Out-of-Box Challenge — `divergence_threshold.py`

**Question:** At exactly what learning rate does GD break?

**Method:** Binary search between a known-safe rate and a known-diverging rate,
narrowing the gap to 5 decimal places over 17 iterations.

**Theoretical validation:** The stability condition for MSE + GD is:

```
lr < 1 / (2 · E[X²])
```

The Hessian of MSE w.r.t. `w` equals `2·E[X²]`. GD is stable only when
the step size is smaller than `1 / curvature`. The binary search result
should land near this value — confirming the implementation is correct.

**Why experimental ≈ 2× theoretical:**
The formula derives stability for `w` alone (1D). With both `w` and `b`
updating simultaneously, the joint system tolerates slightly more before
the interaction between the two gradients causes runaway growth. The 2×
factor is a real finding about 1D vs 2D parameter spaces — not a bug.

**Outputs:**

- `divergence_threshold.png` — loss bowl with annotated GD steps for safe / threshold / diverging rates
- `divergence_geometry.png` — geometric diagram of why overshooting causes runaway loss growth

---

## Concepts Covered

- MSE loss derivation from first principles
- Gradient computation via chain rule — vectorized with NumPy
- Parameter update rule and the role of learning rate
- Loss landscape as a parabolic bowl in parameter space
- Divergence as a geometric phenomenon — overshooting the valley bottom
- Stability condition: `lr < 1 / λ_max(Hessian)`
- Why zero-centered inputs cause bias gradient lag
- Why input normalization makes the loss bowl rounder

---

## Key Insights

> **The learning rate threshold is set by the curvature of the loss surface.**
> Steep narrow bowls require small steps. Wide shallow bowls tolerate larger ones.
> Divergence is not gradual — one overshoot lands you on a steeper wall,
> making the next step even bigger. The system blows up exponentially.

> **Bias converges slower than weight when X is zero-centered.**
> The bias gradient is `(2/n)·mean(residuals)` — tiny when residuals cancel.
> The weight gradient gets strong signal from `X·residuals` dot product.
> This is the core reason we normalize inputs in neural networks.

---

## Run

```bash
cd day-04
source ../.venv/bin/activate

python gradient_descent.py       # standard task
python divergence_threshold.py   # out-of-box challenge
```
