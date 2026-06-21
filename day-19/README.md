# Day 19 — Keras Deep Dive: Architecture Decisions & the Lottery Ticket Hypothesis

## Standard Task

Systematic MLP architecture search on the Adult Census Income dataset:

- Depth: 1–5 hidden layers
- Width: 32–512 neurons per layer
- Dropout: 0, 0.2, 0.5

A total of **75 configurations** are evaluated and logged with:

- Validation accuracy
- Training time
- Parameter count
- Epochs trained

The best-performing configuration is selected automatically using validation accuracy, retrained, and evaluated on a held-out test set.

### Dataset

Adult Census Income Dataset

- Total samples: 32,561
- Features after preprocessing: 104
- Train samples: 23,524
- Validation samples: 4,152
- Test samples: 4,885
- Positive-class rate: ~24.1% across all splits

---

## File Structure

```
    day-19/         ← Keras Deep Dive: Architecture Decisions & the Lottery Ticket Hypothesis
       ├── 01_preprocess_data.py              # Clean Adult dataset, encode, scale, split
       ├── 02_architecture_search.py          # Train 75 architecture configs, log results
       ├── 03_train_best_architecture.py      # Train + save best config, save initial weights
       ├── 04_lottery_ticket_pruning.py       # Lottery ticket + random pruning experiments
       ├── 05_compare_results.py              # Generate comparison visualizations
       ├── README.md                          # Project overview and results
       ├── config.py                          # Centralized paths and hyperparameters
       ├── data
       │   ├── processed
       │   │   └── splits.npz                 # Preprocessed train/val/test splits
       │   └── raw
       │       └── adult.csv                  # Original Adult Census Income dataset
       ├── models
       │   ├── best_model.keras               # Best-performing trained network
       │   ├── best_model_initial_weights.npz # Original initialization for lottery-ticket rewinding
       │   └── preprocessor.joblib            # Saved preprocessing pipeline
       ├── outputs
       │   ├── architecture_search
       │   │   ├── 3d_architecture_grid.png   # 3D architecture search visualization
       │   │   ├── depth_width_heatmaps.png   # Depth-width performance heatmaps
       │   │   └── grid_results.csv           # Results from all 75 architectures
       │   ├── best_model
       │   │   ├── best_config.json           # Selected architecture configuration
       │   │   ├── test_metrics.json          # Final test performance metrics
       │   │   └── training_history.csv       # Epoch-by-epoch training history
       │   └── pruning
       │       ├── comparison.csv             # Full vs lottery-ticket vs random-pruned metrics
       │       └── pruning_comparison.png     # Pruning experiment visualization
       └── utils
           ├── __init__.py
           └── architecture.py                # Shared MLP model builder

    learning-journal/
    └── day-19.md

```

## Out-of-Box Challenge: The Lottery Ticket Hypothesis

Frankle & Carbin (2019) showed that large neural networks contain small,
sparse subnetworks ("winning tickets") that — when reset to their **original**
initial weights and trained in isolation — can match the accuracy of the full
network.

This day tests that claim on the best architecture found above:

1. Train the full model (done in step 3), then magnitude-prune 50% of the
   smallest weights in every Dense kernel.
2. Rewind the surviving weights to their pre-training random initialization
   (not a fresh reinit), apply the mask, and retrain from scratch.
3. **Control:** repeat with a random mask at identical sparsity, same initial
   weights — to confirm magnitude pruning is doing something non-trivial
   rather than sparsity alone explaining any result.
4. Compare full vs. lottery-ticket vs. random-pruned on test accuracy and
   training time.

---

## Pipeline

| Step | Script                          | Purpose                                                        |
| ---- | ------------------------------- | -------------------------------------------------------------- |
| 0    | `config.py`                     | All paths, hyperparameters, grid definitions                   |
| 1    | `01_preprocess_data.py`         | Clean Adult dataset, encode, scale, split                      |
| 2    | `02_architecture_search.py`     | Train 75 architecture configs, log results                     |
| 3    | `03_train_best_architecture.py` | Train + save best config, save its initial weights             |
| 4    | `04_lottery_ticket_pruning.py`  | Magnitude pruning + lottery ticket retraining + random control |
| 5    | `05_compare_results.py`         | Generate all comparison plots                                  |

Shared model-builder: `utils/architecture.py`

---

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

Download `adult.csv` from Kaggle:

https://www.kaggle.com/datasets/uciml/adult-census-income

Place it at:

```text
data/raw/adult.csv
```

---

## Run Order

```bash
python 01_preprocess_data.py
python 02_architecture_search.py
python 03_train_best_architecture.py
python 04_lottery_ticket_pruning.py
python 05_compare_results.py
```

---

## Best Architecture

Selected automatically from architecture search:

| Hyperparameter | Value |
| -------------- | ----- |
| Depth          | 5     |
| Width          | 512   |
| Dropout        | 0.2   |

---

## Final Model Performance

| Metric               | Value  |
| -------------------- | ------ |
| Test Accuracy        | 0.8545 |
| Test Loss            | 0.3188 |
| Early Stopping Epoch | 10     |
| Input Features       | 104    |

Observations:

- Validation accuracy peaked early.
- Additional epochs produced diminishing returns.
- The model achieved strong performance without requiring all 60 training epochs.

---

## Lottery Ticket Results

Pruning fraction: **50%**

| Model          | Test Accuracy | Epochs | Sparsity |
| -------------- | ------------- | ------ | -------- |
| Full Model     | 0.8545        | N/A    | 0.000    |
| Lottery Ticket | 0.8547        | 11     | 0.499    |
| Random Pruned  | 0.8542        | 13     | 0.499    |

### Result Summary

- Lottery-ticket retraining slightly exceeded the full model.
- Random pruning achieved nearly identical performance.
- At 50% sparsity, the network retains enough redundancy that both pruning approaches remain highly effective.
- Magnitude pruning provided only a marginal advantage over random pruning on this dataset.

---

## Generated Outputs

### Processed Data

```
data/processed/splits.npz
```

### Architecture Search

```
outputs/architecture_search/grid_results.csv
outputs/architecture_search/3d_architecture_grid.png
outputs/architecture_search/depth_width_heatmaps.png
```

### Best Model

```
outputs/best_model/best_config.json
outputs/best_model/training_history.csv
outputs/best_model/test_metrics.json
```

### Pruning Results

```
outputs/pruning/comparison.csv
outputs/pruning/pruning_comparison.png
```

---

## Saved Models

```
models/preprocessor.joblib
models/best_model.keras
models/best_model_initial_weights.npz
```

---

## Key Findings

### Architecture Search

- Deeper and wider networks consistently outperformed smaller architectures.
- The best-performing configuration was the largest architecture evaluated:
  - Depth = 5
  - Width = 512
  - Dropout = 0.2

### Generalization

- Validation and test performance remained closely aligned.
- No severe overfitting was observed despite the model size.

### Lottery Ticket Experiment

- 50% of weights could be removed while preserving accuracy.
- Lottery-ticket retraining matched full-model performance.
- Random pruning produced nearly identical results, suggesting substantial parameter redundancy.

### Main Takeaway

This experiment demonstrates that a large fraction of parameters in the trained network are unnecessary for maintaining predictive performance on the Adult Census Income dataset. At moderate sparsity levels, both structured magnitude pruning and random pruning preserve accuracy remarkably well.
