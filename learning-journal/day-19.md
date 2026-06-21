# Day 19 — June 21, 2026

## What I built today:

Built a complete neural network experimentation pipeline using TensorFlow/Keras on the Adult Census Income dataset.

The project included:

- Data preprocessing pipeline with:
  - Missing value handling
  - Numerical feature scaling
  - One-hot encoding of categorical variables
  - Reproducible train/validation/test splits

- Systematic architecture search across:
  - Depths: 1–5 hidden layers
  - Widths: 32–512 neurons
  - Dropout rates: 0.0, 0.2, 0.5

- Automated experiment tracking that logged:
  - Validation accuracy
  - Training accuracy
  - Parameter count
  - Training time
  - Epochs trained

- Best-model training pipeline that:
  - Selected the highest-validation-accuracy architecture
  - Saved the trained model
  - Saved original random initialization weights for later lottery-ticket experiments

- Lottery Ticket Hypothesis implementation:
  - Magnitude-based pruning
  - Weight rewinding to original initialization
  - Sparse training with mask enforcement after every batch
  - Random-pruning control experiment

- Visualization suite generating:
  - 3D architecture-search scatter plots
  - Depth-vs-width heatmaps
  - Pruning comparison charts

Additionally, I reorganized the project structure and moved processed datasets from the outputs directory into the input/data workflow for cleaner separation between generated artifacts and reusable datasets.

### Dataset Summary

- Raw samples: 32,561
- Train samples: 23,524
- Validation samples: 4,152
- Test samples: 4,885
- Processed feature dimension: 104

### Best Architecture Found

- Depth: 5 hidden layers
- Width: 512 neurons
- Dropout: 0.2

### Full Model Performance

- Test Accuracy: 0.8545
- Test Loss: 0.3188

---

## The out-of-box challenge result:

The Lottery Ticket Hypothesis was successfully reproduced on this dataset.

### Comparison

| Model          | Test Accuracy | Epochs | Sparsity |
| -------------- | ------------- | ------ | -------- |
| Full Model     | 0.8545        | N/A    | 0.000    |
| Lottery Ticket | 0.8547        | 11     | 0.499    |
| Random Pruned  | 0.8542        | 13     | 0.499    |

### Key observations

- The lottery-ticket subnetwork slightly outperformed the full model:
  - 0.8547 vs 0.8545 accuracy
  - Improvement: +0.0002

- The lottery-ticket model also slightly outperformed the random-pruned control:
  - 0.8547 vs 0.8542 accuracy
  - Improvement: +0.0004

- Both sparse models preserved approximately 50% sparsity.

- The lottery-ticket model converged in fewer epochs:
  - Lottery ticket: 11 epochs
  - Random-pruned: 13 epochs

- Training time:
  - Lottery ticket: 7.6 seconds
  - Random-pruned: 8.8 seconds

### Conclusion

On the Adult Census dataset, a carefully selected sparse subnetwork was able to match—and very slightly exceed—the performance of the fully trained dense model while retaining only about half of the weights.

---

## What surprised me:

Several results were unexpected:

1. Removing roughly 50% of the network weights produced virtually no loss in test accuracy.

2. The lottery-ticket network achieved marginally better accuracy than the original full model despite having far fewer active parameters.

3. The random-pruned control performed almost as well as the lottery ticket, suggesting that this dataset may not be difficult enough to expose large differences between intelligent pruning and random pruning.

4. The architecture search selected the largest architecture tested (5 layers × 512 neurons) even though the Adult dataset is relatively small.

5. Early stopping terminated training very quickly despite allowing up to 60 epochs.

---

## What I don't fully understand yet:

1. Why the lottery-ticket advantage over random pruning was so small on this dataset.

2. Whether larger datasets would produce a much larger gap between magnitude pruning and random pruning.

3. How sparsity levels beyond 50% (70%, 80%, 90%, etc.) affect performance.

4. Whether iterative pruning would outperform the single-shot pruning approach used here.

5. How lottery-ticket behavior changes in deeper architectures such as CNNs, ResNets, Transformers, and modern LLMs.

6. Why the architecture search consistently favored the largest network tested instead of a smaller model with comparable generalization.

---

## GitHub commit made: ✅

`Day 19 — Keras Deep Dive: Architecture Decisions & the Lottery Ticket Hypothesis`

## Tomorrow's priority:

Day 20 — Convolutional Neural Networks (CNNs) for image classification. Move beyond tabular data and implement CNN architectures on image datasets, compare fully connected networks against convolutional networks, analyze parameter efficiency, visualize learned filters and feature maps, and investigate why spatial inductive bias makes CNNs outperform traditional MLPs on vision tasks.
