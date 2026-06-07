# Day 04 — Learning Journal

**Date:** [fill in today's date]
**Phase:** Foundation — NumPy Linear Algebra & Gradient Descent

---

## What I built today

**Standard task:** Batch gradient descent from scratch for linear regression —
no sklearn, no scipy, only NumPy. Generated synthetic data (`y = 3x + 7 + noise`)
where I knew the true answer, implemented MSE loss and gradients by hand,
ran 10,000 epochs with three different learning rates, and confirmed GD
recovers parameters close to the truth.

Built three diagnostic plots: loss curves for all three learning rates side
by side, the GD-fitted line overlaid on the data scatter, and residual
diagnostics (scatter + histogram).

**Out-of-box challenge:** Binary search to find the exact learning rate where
GD diverges experimentally, then validated the result against the theoretical
stability formula `lr < 1 / (2·E[X²])`. Built a loss landscape plot with
annotated GD steps and a geometric diagram explaining why divergence happens.

---

## Concepts I understood before writing code

**The loss function creates a bowl.** MSE gives you a parabolic surface in
parameter space. The bottom of that bowl is the minimum loss — the best
parameters. GD's only job is to find the bottom.

**The gradient is your compass.** At any point on the bowl, the gradient
points uphill. You step in the opposite direction. The update rule is simply:
`w ← w - lr · ∂L/∂w`. Repeat until you reach the bottom.

**The learning rate controls step size — and it is everything.**
Too small: you crawl and never converge in reasonable time.
Just right: smooth descent, reaches the minimum efficiently.
Too large: you overshoot the bottom, land on the opposite wall which is
steeper, next step is even bigger, loss explodes to infinity. That is
divergence — and it is sudden and catastrophic, not gradual.

**Divergence is a geometric phenomenon.** The loss bowl has a width
determined by its curvature (the Hessian). If your step size exceeds that
width, you overshoot every single time, landing further away on each step.
The stability condition is `lr < 1 / (2·E[X²])` — step size must be
smaller than `1 / curvature`.

---

## Out-of-box challenge result

Binary search found the divergence threshold at `lr ≈ 0.126` (17 iterations,
precision to 5 decimal places).

Theoretical prediction: `1 / (2 × E[X²]) = 1 / (2 × 8.417) ≈ 0.059`.

The experimental value is roughly 2× the theoretical. This is not an error —
the formula derives stability for `w` alone in a 1D system. With both `w` and
`b` updating simultaneously, the joint system tolerates slightly more before
their interaction causes runaway growth. The 2× gap is a real finding about
the difference between 1D and 2D parameter spaces.

---

## What surprised me

**Bias converges much slower than weight.** With X zero-centered (mean ≈ 0),
the bias gradient `(2/n)·mean(residuals)` is tiny because positive and
negative residuals partially cancel. The weight gradient gets strong signal
from `X·residuals` dot product. After 10,000 epochs, `w` converged to 3.011
(error: 0.011) but `b` only reached 6.933 (error: 0.067). This is not a bug —
it directly demonstrates why input normalization matters in neural networks.
Un-normalized features create elongated bowls with very different curvatures
in different directions, forcing a tiny learning rate that makes convergence
slow everywhere.

**The residual plot is genuinely informative.** Random scatter around zero
means the model captured the linear signal and only noise remains. A pattern
in residuals would mean the model is missing something systematic. Seeing it
visually — not just reading about it — makes it stick.

---

## What I don't fully understand yet

- The derivation of `1 / (2·E[X²])` from eigenvalue conditions on the
  Hessian. I can verify it numerically but want to work through the full
  linear algebra derivation.

- Mini-batch GD vs batch GD: how does gradient noise in mini-batch affect
  the stability threshold? Presumably it lowers it further since noisy
  gradients add extra instability.

- How the loss bowl shape changes in higher dimensions (many features).
  Each feature adds a dimension — what does "curvature" mean across all
  of them simultaneously, and how does normalization fix the elongation?

---

## Debugging lessons from today

- `KeyError` on a dict lookup almost always means the key string does not
  match exactly — check for invisible differences like spaces around `=`.
- `plt.show()` in a script can crash silently on macOS and kill saves that
  come after it. Use `plt.close()` instead when running `.py` files.
- Format specifier `:.1f` uses digit `1`, not letter `l`. They look identical
  in most fonts.
- Missing `()` on a method call — `y.max` vs `y.max()` — gives `TypeError`
  about format strings, not an obvious attribute error.

---

## GitHub commit made: ✅

## Commit message: `day-04: gradient descent from scratch + divergence threshold study`

## Tomorrow's priority:

Day 05 — Pandas EDA on Titanic. Read the out-of-box challenge carefully
before touching the dataset: find one finding that no standard Titanic
analysis covers.
