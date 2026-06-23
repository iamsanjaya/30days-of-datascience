# Day 20 — June 22, 2026

## What I built today

Built a complete CNN training and regularization experimentation pipeline on CIFAR-10 to study overfitting behavior, regularization techniques, and learning-rate selection.

The project included:

- Reproducible CIFAR-10 data-loading pipeline using PNG images
- Class-balance verification and dataset inspection
- Fixed-seed train/validation overfit subset generation
- Deliberately oversized CNN designed to overfit a small dataset
- Modular CNN builder with optional:
  - Dropout
  - L2 regularization
  - Batch Normalization
  - Data augmentation
  - Early stopping

- Automated experiment runner comparing multiple regularization strategies
- Learning Rate Range Test implementation based on Leslie Smith (2017)
- Visualization suite generating:
  - Class balance plots
  - Overfitting curves
  - Regularization comparison grids
  - LR Range Test plots

### 01_data_preparation.py

Verified CIFAR-10 download integrity and dataset structure.

Implemented reproducible train/validation subset generation that:

- Produces the same samples every run
- Maintains class balance
- Prevents train/validation leakage
- Creates a deliberately small dataset suitable for studying overfitting

### 02_overfit_baseline.py

Trained a large unregularized CNN on a 500-image training subset.

Implemented:

- Reduced learning rate (`1e-4`)
- Gradient clipping (`clipnorm=1.0`)
- History persistence
- Model checkpoint saving

Observed a textbook overfitting pattern:

- Train accuracy: 96.6%
- Validation accuracy: 31.0%
- Generalization gap: 65.6%

Training performance continued improving while validation performance steadily deteriorated.

### 03_regularization_comparison.py

Implemented and compared six regularization strategies:

1. Dropout
2. L2 Regularization
3. Batch Normalization
4. Data Augmentation
5. Early Stopping
6. Combined Configuration

Added post-training evaluation logic to correctly measure restored early-stopping models after discovering that Keras history metrics do not necessarily represent restored weights.

### 04_lr_range_test.py

Implemented Leslie Smith's Learning Rate Range Test.

Key implementation details:

- Exponential LR ramp from `1e-7` to `10`
- Full CIFAR-10 training pool
- One-epoch sweep
- Automatic landmark detection:
  - `lr_start_decreasing`
  - `lr_steepest_descent`
  - `lr_diverge`

Cross-validated results across multiple runs to measure stability under TensorFlow Metal's non-deterministic GPU execution.

---

## The out-of-box challenge result

### LR Range Test landmarks

Run 1:

- lr_start_decreasing: `1.00e-05`
- lr_steepest_descent: `2.73e-04`
- lr_diverge: `5.88`

Run 2:

- lr_start_decreasing: `7.73e-05`
- lr_steepest_descent: `3.17e-04`
- lr_diverge: `9.43`

Most important observation:

- `lr_steepest_descent` remained stable near `3e-4` across both runs despite variation in the other landmarks.

### How `lr_steepest_descent` compared to the Day 19 manually-tuned LR

Day 19 used:

- Learning rate: `1e-3`

LR Range Test recommendation:

- Approximately `3e-4`

The automatically discovered learning rate is roughly three times smaller than the manually selected Day 19 value.

This result was surprising because it suggests the architecture search on Day 19 succeeded despite operating at a learning rate somewhat higher than the steepest-loss-descent region identified by the LR Range Test.

The difference is likely explained by:

- Different datasets (Adult vs CIFAR-10)
- Different architectures (MLP vs CNN)
- Different optimization landscapes
- Early stopping limiting the impact of a suboptimal learning rate

More importantly, the LR Range Test demonstrated that learning rates can be selected empirically rather than through repeated trial-and-error.

---

## What surprised me

1. Batch Normalization produced the highest validation accuracy of all individual regularization techniques.

2. BatchNorm reached nearly 100% training accuracy within a few epochs while validation performance initially worsened before recovering.

3. Data augmentation performed significantly worse than expected and eventually destabilized training despite being one of the most commonly recommended regularization methods.

4. The smallest train/validation gap belonged to the augmentation configuration, but this occurred because both metrics became poor rather than because the model generalized well.

5. Combining every regularization technique produced the worst overall result.

6. The failure of the combined configuration was not caused by overfitting but by Early Stopping restoring weights from epoch 1 before the heavily regularized model had enough time to learn.

7. The LR Range Test produced remarkably stable `lr_steepest_descent` estimates despite TensorFlow Metal's non-deterministic execution.

8. Several difficult bugs were caused not by model design but by TensorFlow Metal and Keras implementation details, including optimizer behavior, BatchNorm graph execution issues, and metric-reporting subtleties.

---

## What I don't fully understand yet

1. Why BatchNorm outperformed Dropout and L2 regularization so clearly on this specific architecture.

2. Why augmentation destabilized training instead of improving generalization.

3. Whether augmentation would behave normally if BatchNorm were enabled simultaneously.

4. How BatchNorm's running statistics evolve during the first few epochs and produce the temporary validation-performance collapse observed here.

5. How much of the observed behavior is specific to small datasets versus general CNN training dynamics.

6. Whether increasing patience from 8 to 30 would completely rescue the combined configuration.

7. How LR Range Test recommendations compare to optimal learning rates across many architectures and datasets.

8. How these regularization interactions change in modern architectures such as ResNets, Vision Transformers, and large-scale foundation models.

---

## GitHub commit made: ✅

`Day 20 — Training Dynamics: Diagnosing an Overfit CNN`

## Tomorrow's priority

Day 21 — Transfer Learning and Modern CNN Architectures.

Move beyond training CNNs from scratch and investigate pretrained networks, feature extraction versus fine-tuning, transfer-learning workflows, parameter efficiency, and how modern architectures leverage previously learned visual representations to achieve strong performance with limited labeled data.
