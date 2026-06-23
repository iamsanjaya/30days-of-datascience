# Day 20 — Training Dynamics: Diagnosing an Overfit CNN

Phase 3 (Deep Learning). Standard task: deliberately overfit a CNN on CIFAR-10,
then fix it with Dropout, L2, BatchNorm, augmentation, and early stopping.
Out-of-box challenge: implement the LR Range Test (Leslie Smith, 2017).

**Status: complete.** All four scripts ran successfully end-to-end on
CIFAR-10 (Apple M4 / tensorflow-metal). `diagnostic_report.md` is filled in
with the actual results. See the table below for the issues hit along the
way and how they were resolved — useful if you re-run this on different
hardware.

## Dataset

Download **["CIFAR-10 PNGs in folders"](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders)**
from Kaggle and unzip it so the layout looks like this:

```
day-20/data/raw/cifar10/
├── train/
│   ├── airplane/*.png
│   ├── automobile/*.png
│   ├── bird/*.png
│   ├── cat/*.png
│   ├── deer/*.png
│   ├── dog/*.png
│   ├── frog/*.png
│   ├── horse/*.png
│   ├── ship/*.png
│   └── truck/*.png
└── test/
    └── (same 10 class subfolders)
```

## Project structure

```
day-20/
├── config.py                        # paths, hyperparameters, regularization configs
├── 01_data_preparation.py           # verify download, class balance, overfit subset check
├── 02_overfit_baseline.py           # train the deliberately oversized, unregularized CNN
├── 03_regularization_comparison.py  # Dropout / L2 / BatchNorm / Augmentation / EarlyStop / combined
├── 04_lr_range_test.py              # out-of-box: LR Range Test
├── utils/
│   ├── __init__.py
│   ├── data.py                      # PNG loading, reproducible stratified subsetting
│   ├── architecture.py              # overfit baseline + toggleable regularized CNN
│   ├── training.py                  # compile/fit helpers, history I/O, LR range test callback
│   └── visualization.py             # training-curve grids, LR range test plot
├── data/raw/cifar10/                # gitignored — place the Kaggle download here
├── outputs/                         # gitignored — plots + JSON histories land here
│   └── histories/                   # per-config training history, saved as JSON
├── models/                          # gitignored — saved .keras model files
└── diagnostic_report.md             # filled in with actual results

learning-journal/
    └── day-20.md
```

## Run order

```bash
python 01_data_preparation.py
python 02_overfit_baseline.py
python 03_regularization_comparison.py
python 04_lr_range_test.py
```

`02` must run before `03` — the regularization comparison reuses the
baseline's saved history (`outputs/histories/baseline.json`) instead of
retraining it.

## What gets generated

- `outputs/class_balance.png` — train/test class counts
- `outputs/overfit_baseline.png` — the textbook overfitting signature
- `outputs/regularization_comparison.png` — all configs side by side
- `outputs/lr_range_test.png` — loss vs LR with the three landmark points annotated
- `outputs/histories/*.json` — raw per-epoch metrics for every config
- `models/overfit_baseline.keras` — the unregularized baseline model

## Issues hit during the real run, and the fixes applied

This list exists because every one of these was a genuine bug or platform
quirk hit while actually running this on Apple Silicon — not hypothetical.

| #   | Issue                                                                                                                                                                                                                                                                                                                         | Fix                                                                                                                                                                                                               |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `02`'s baseline diverged numerically (loss → billions by epoch 100) instead of overfitting cleanly. Root cause: 4 conv layers + a 16.7M-param dense layer, zero normalization, constant `Adam(lr=1e-3)` for 100 epochs — weight magnitudes drift and Adam's adaptive scaling compounds rather than damps it.                  | Added `config.BASELINE_LEARNING_RATE = 1e-4` and `config.BASELINE_CLIPNORM = 1.0`, used by `02` and `03` only — **not** by `04`, which needs to observe genuine divergence at high LR.                            |
| 2   | `batchnorm` config crashed the process outright: `Mutation::Apply error... fanout exists for missing node`, an abort signal from the Metal graph remapper. The default `tf.keras.optimizers.Adam`'s op pattern, combined with BatchNorm's extra moving-mean/variance update ops, confuses tensorflow-metal's graph optimizer. | Switched `training.compile_model` to `tf.keras.optimizers.legacy.Adam` — aligns with TensorFlow recommendations on Apple Silicon and avoids Metal graph remapper issues.                                          |
| 3   | Terminal flooded with `stateless_random_op... GPU implementation does not produce the same series as CPU` — one line per random op call, drowning out training logs during the `augmentation` config.                                                                                                                         | Set `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` at the top of `config.py` before importing TensorFlow.                                                                                                   |
| 4   | `early_stopping`/`combined` configs use `restore_best_weights=True`, but reporting used `history.history[...][-1]` (last epoch metrics) instead of restored best weights. This made results misleading for comparison.                                                                                                        | Updated `03_regularization_comparison.py` to run `model.evaluate(..., return_dict=True)` after `fit()` for all early-stopping runs.                                                                               |
| 5   | Multiple Pylance errors (`reportOptionalMemberAccess`, `reportMissingModuleSource`, `reportArgumentType`, `reportIndexIssue`) caused by TensorFlow/Keras type stubs not matching runtime behavior.                                                                                                                            | Introduced strict typing workarounds: `assert` guards for `self.model`, switched to `tf.keras.*` imports, explicit `Dict[str, Any]` / `List[Callback]`, and `typing.cast` for `model.evaluate(return_dict=True)`. |

## Why the overfit subset is fixed and disjoint

`utils/data.get_overfit_subset()` first splits the full training pool into
disjoint train/val partitions (per class, before any subsampling), then
draws a fixed-seed stratified sample from each. This guarantees:

- the same 500 train / 200 val images every run (reproducible across scripts)
- zero overlap between train and val (no leakage inflating val accuracy)

## A note on the LR Range Test ramp

The roadmap describes "linearly increasing LR from 1e-7 to 10." Implemented
literally, that would spend nearly every batch near 10 and barely sample the
low end — 1e-7 to 10 spans eight orders of magnitude. `LRRangeTestCallback`
ramps the LR **exponentially** (constant multiplicative step per batch),
which is what Leslie Smith's original technique actually does.

`04` also uses `math.ceil` (not `//`) when computing the expected number of
batches, so the per-step LR multiplier matches the actual number of steps
Keras runs (40,000 images / 128 batch size = 312.5 → 313 actual steps,
including the partial last batch).

No random seed is set before model construction in `04`, and the Metal GPU
backend is non-deterministic by TensorFlow's own admission — so re-running
`04` gives slightly different `lr_start_decreasing`/`lr_diverge` values each
time (these are noisy edge-detection points on the loss curve). The
`lr_steepest_descent` value — the one that actually matters for picking a
training LR — was stable at ~3e-4 across two separate runs. If you want
bit-for-bit reproducibility on all three landmarks, add
`tf.random.set_seed(config.RANDOM_SEED)` before `architecture.build_regularized_cnn(...)`
in `04`.
