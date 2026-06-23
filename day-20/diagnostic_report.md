# Day 20 — Diagnostic Report: Regularization on an Overfit CNN

All numbers below are from the corrected run (post `model.evaluate()` fix for
early-stopping configs — see `03_regularization_comparison.py`). Training
used `BASELINE_LEARNING_RATE=1e-4` + `clipnorm=1.0` for every config, on the
identical 500-train/200-val stratified subset.

## 1. Baseline — what overfitting looks like

- Final train accuracy: **0.966** | Final val accuracy: **0.310** | Gap: **0.656**
- Loss curve: train loss drops below 1.0 by ~epoch 15-18 and continues
  falling to ~0.4-0.9 (noisy but trending down) through epoch 100. val_loss
  rises almost from epoch 1 and never turns back — by epoch 100 it reaches
  **58.9**, roughly 85x the final train loss.
- Accuracy curve: train accuracy crosses 90% around epoch 40 and reaches
  96.6% by epoch 100. val_accuracy peaks in the mid-40s% somewhere around
  epoch 25-40 (noisy, e.g. 0.49 at epoch 29) then drifts back down to 0.31
  by epoch 100 — the model doesn't just plateau on val, it gets _worse_ the
  longer training continues past its useful point.
- This is the textbook shape: with this fix-free architecture, divergence
  between train and val starts almost immediately (by epoch ~10) and widens
  monotonically for the rest of training.

## 2. Dropout

- Final train/val gap: **0.578** (baseline was 0.656) — an 0.078 absolute
  reduction, about 12% relative.
- Curve shape: train accuracy growth is slower and noisier than baseline's
  — Dropout's per-batch random unit masking adds variance to the training
  signal itself, visible as a much bumpier train_acc curve (e.g. epoch 50:
  0.92, epoch 51: 0.898, epoch 52: 0.92 — oscillating rather than
  monotonic). val_loss still climbs steadily (reaching ~32 by epoch 100),
  just less steeply than baseline's 58.9.
- New behavior introduced: yes — visibly noisier train metrics throughout,
  which is Dropout doing exactly what it's designed to do (preventing the
  network from settling into a single fixed memorization pattern).

## 3. L2 Regularization

- Final train/val gap: **0.544** — similar accuracy-gap reduction to
  Dropout, but a very different loss story.
- val_loss reached **235** by epoch 100 — roughly 7x higher than Dropout's
  val_loss at the same point, despite a _similar_ val_accuracy (0.350 vs
  0.360). This is the key difference in curve shape: L2 constrains weight
  _magnitude_, not output _confidence_. A wrongly-classified example can
  still get an extremely confident (and therefore high-loss) wrong
  prediction under L2, whereas Dropout's training-time noise tends to keep
  the model less overconfident in general. Accuracy and loss tell different
  stories here — accuracy alone would make L2 look comparable to Dropout;
  the loss curve reveals it isn't.
- L2 did not close the gap as well as BatchNorm, and its loss-curve
  instability (large epoch-to-epoch swings, e.g. epoch 81: 4.9, epoch 82:
  8.0, epoch 83: 7.7) suggests it's the least numerically stable of the
  three "soft" fixes (Dropout/L2/BatchNorm) on this architecture.

## 4. BatchNorm

- Final train/val gap: **0.515** — smallest gap among the individual
  fixes, and the highest val_accuracy of all seven configs: **0.485**.
- Distinctive curve shape, reproduced identically across two separate runs:
  train accuracy hits 100% by epoch ~5-6 (fastest of any config). val_loss
  then _spikes_ to ~5.4 around epoch 11-13 — worse than at epoch 1 — before
  steadily _descending_ back down to ~1.75-1.8 by epoch 45 and plateauing
  there for the remaining ~55 epochs, with val_accuracy climbing in lockstep
  from ~0.10 up to ~0.48.
- Why this shape happens: BatchNorm uses per-batch statistics during
  training but a running (momentum-averaged) mean/variance at inference.
  With only 16 batches per epoch, those running statistics are highly
  unreliable in the first ~10 epochs — by the time train accuracy hits
  100%, the running statistics haven't caught up, so validation (which
  uses them) performs poorly or even gets briefly worse. As more epochs
  pass, the running average converges toward the true population
  statistics and validation performance recovers and stabilizes. BatchNorm
  wasn't designed as a regularizer, but on this setup it produced the best
  generalization of any single fix — likely because normalizing activations
  also smooths the loss landscape Adam is optimizing over, independent of
  any explicit regularization effect.

## 5. Data Augmentation

- Final train/val gap: **0.026** — looks like the best result by this one
  number. It is not. Train accuracy _fell_ to **0.366**, the lowest of any
  config — a small gap here means both numbers are bad, not that the model
  generalizes well.
- Curve shape: relatively stable for the first ~20-40 epochs (train acc
  climbing slowly to ~0.50-0.53), then destabilizes — loss climbs from
  single digits into the 40s-70s range by epoch 100, with train accuracy
  drifting back down into the 0.30s.
- Reproduced identically across two full reruns, so this is a real property
  of this config, not a fluke. This is the only config that adds stochastic
  per-batch input transformations (flip/rotation/zoom/translation) without
  also adding BatchNorm — and it's the only individual fix that destabilized
  rather than improved on baseline. The likely mechanism: augmentation
  increases the effective variability of activations batch-to-batch, and
  with nothing in the architecture to normalize those activations, that
  variability compounds over many epochs into the same kind of numerical
  drift that the LR/clipnorm fix solved for the _plain_ baseline — just
  reintroduced by a different source.

## 6. Early Stopping

- Stopped at epoch 19 (best `val_loss` was 1.8359 at epoch 11; patience=8
  with no improvement in epochs 12-19 triggered the stop).
- Restored-model metrics (via `model.evaluate()`, not the last logged
  epoch): train_acc **0.620**, val_acc **0.410** — second-best val_accuracy
  of all seven configs, despite training the fewest effective epochs.
- One subtlety worth flagging: the _logged_ epoch-11 train accuracy was
  0.586, not 0.620. That's not an error — Keras's logged per-epoch training
  accuracy is a running average computed _while weights are still updating_
  within that epoch, whereas `model.evaluate()` afterward measures the
  fixed, final end-of-epoch weights in one deterministic pass. The
  evaluated number is usually slightly higher because it reflects only the
  improved end-of-epoch state, not the average across the epoch's
  in-progress updates.
- Early stopping doesn't fix overfitting — it just stops before the gap
  gets as bad as baseline's. The curve supports that framing exactly:
  epochs 12-19 show val_loss actively climbing (1.93 -> 3.00) while it
  waited out its patience window; the _fix_ here is timing, not mechanism.

## 7. Combined (all fixes together)

- Restored-model metrics: train_acc **0.116**, val_acc **0.110** — both
  near the 1-in-10 random-guess baseline for 10 classes. Gap: 0.006 — by
  far the smallest gap of any config, and the worst outcome of any config.
- This is not the same story as "Dropout and BatchNorm fighting each
  other" (a real phenomenon in the literature, but not what's visible
  here). The actual mechanism: `combined`'s best `val_loss` occurred at
  **epoch 1** (5.7759) and was never beaten again — patience=8 then fired
  almost immediately, stopping training at epoch 9 and restoring back to
  essentially the random-initialization state.
- Why epoch 1 was already "best": stacking four regularizers (Dropout, L2,
  BatchNorm, Augmentation) simultaneously slows convergence dramatically —
  Dropout and L2 alone took 80-90+ epochs each to reach their own best
  train accuracy on this same data, far slower than baseline's. With all
  of them compounding, the model needs far more than 8 epochs of patience
  to show any sustained improvement at all.
- This produces a falsifiable, testable claim rather than a vague
  "regularizers interact badly" statement: **`EARLY_STOPPING_PATIENCE=8`
  was tuned for single fixes, not for a config combining four of them.**
  Rerunning `combined` with a much higher patience (e.g. 30) would
  distinguish between two hypotheses — if it then trains and eventually
  overfits like the individual fixes did, the problem was patience, not
  the regularizers' interaction. If it stays flat regardless, that would
  support genuine optimization interference between Dropout and BatchNorm.
  This wasn't tested as part of this run.

## 8. Ranking by val_accuracy

1. **BatchNorm** — 0.485 (clear best; fast convergence + eventual stability)
2. **Early Stopping** — 0.410 (best timing-based fix; cheapest to apply)
3. **Dropout** — 0.360
4. **L2** — 0.350 (similar accuracy to Dropout, much worse loss behavior)
5. **Augmentation** — 0.340 (driven by collapse, not generalization — see §5)
6. **Baseline** — 0.310
7. **Combined** — 0.110 (worse than doing nothing — see §7)

This ranking is specific to this dataset size (500 train images), this
architecture (4 conv layers, ~18.8M params, no normalization unless BatchNorm
is explicitly toggled on), and `EARLY_STOPPING_PATIENCE=8`. It is not a
general claim that BatchNorm beats Dropout, or that augmentation is
unreliable — both conclusions are artifacts of this specific setup,
especially the missing normalization elsewhere in the augmentation case and
the short patience window in the combined case.

## 9. LR Range Test cross-check (out-of-box challenge)

Run twice (architecture: BatchNorm-enabled, full 40,000-image train pool,
one epoch, exponential LR ramp from 1e-7 to 10):

| landmark              | run 1    | run 2    |
| --------------------- | -------- | -------- |
| `lr_start_decreasing` | 1.00e-05 | 7.73e-05 |
| `lr_steepest_descent` | 2.73e-04 | 3.17e-04 |
| `lr_diverge`          | 5.88     | 9.43     |

- `lr_steepest_descent` — the number that actually matters for choosing a
  training LR — is stable across both runs, landing in the same ~3e-4
  neighborhood both times.
- The other two landmarks moved more between runs. No random seed is set
  before model construction in `04_lr_range_test.py`, and the Metal GPU
  backend is explicitly non-deterministic (TensorFlow's own log: "GPU
  implementation does not produce the same series as CPU implementation"),
  so different initial weights each run shift exactly where the noisy early
  loss curve happens to dip (`lr_start_decreasing`) or spike
  (`lr_diverge`) — both edge-detection points on a single noisy curve,
  versus `lr_steepest_descent`, which is the midpoint of a long, consistent
  downward slope and therefore far less sensitive to initialization noise.
- **Comparison against the Day 19 manually-tuned learning rate:** _fill in
  here — record whatever LR you landed on during Day 19's architecture
  search, and note whether ~3e-4 matches, is close, or is way off, with your
  best guess for why._
