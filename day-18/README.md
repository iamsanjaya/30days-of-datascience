# Day 18 — Neural Networks from Scratch + Weight Visualization Experiment

## File Structure

```
day-18/
├── 01_numpy_nn_scratch.py                  # 2-layer NN (784→128 ReLU→10 softmax) built from scratch in
│                                           # NumPy: forward pass, backprop, mini-batch GD on MNIST
├── 02_keras_nn_replica.py                  # Identical architecture in Keras (plain SGD, same lr/epochs)
│                                           # to confirm the from-scratch math matches a framework
├── 03_weight_visualization_experiment.py   # Trains a 2nd model on shuffled labels (real images,
│                                          # scrambled targets); visualizes hidden-layer weights from
│                                          # both models reshaped to 28x28 (real vs. memorized)
├── 04_linear_softmax_templates.py         # Bonus: no-hidden-layer softmax model (784→10 direct).
│                                          # Output weights reshape into recognizable digit templates —
│                                         # contrasts with 03's uninterpretable MLP hidden-layer weights
├── config.py                             # Paths, architecture sizes, hyperparameters for all 4 scripts
├── README.md                             # Day-18 summary, run order, dataset source, results
├── data/
│   └── raw/
│       └── train.csv                      # Kaggle Digit Recognizer (real MNIST, CSV format)
├── models/                                # gitignored — trained weights/params, consumed across scripts
│   ├── numpy_nn_params.npz                # from 01_numpy_nn_scratch.py
│   ├── keras_nn_replica.keras             # from 02_keras_nn_replica.py , loaded by 03 for the "real labels" comparison
│   └── keras_nn_random_label.keras        # from 03_weight_visualization_experiment.py
│
├── outputs/                               # tracked — figures + metrics meant for review
│   ├── metrics/
│   │   ├── numpy_nn_history.csv          # per-epoch loss/test-acc from 01
│   │   └── keras_nn_history.csv           # per-epoch loss/acc/val from 02
│   └── plots/
│       ├── weights_real_labels.png       # 03 — hidden-layer weights, real-label model
│       └── weights_shuffled_labels.png   # 03 — hidden-layer weights, shuffled-label model
│
└── utils/
    ├── __init__.py
    ├── data_loader.py                  # load_mnist_csv(), one_hot_encode()
    ├── preprocess.py                   # prepare_data() — shared train/test prep for 03 and 04
    └── nn_utils.py                     # forward/backward pass, activations, loss — used only by 01

learning-journal/
└── day-18.md
```

## Standard Task

A 2-layer neural network (784 → 128 ReLU → 10 softmax) implemented two ways:

### 1. NumPy Implementation (`01_numpy_nn_scratch.py`)

- Forward pass (ReLU, softmax)
- Cross-entropy loss
- Backpropagation from scratch
- Mini-batch gradient descent on MNIST (CSV format)
- Final accuracy: ~0.8721

### 2. Keras Replica (`02_keras_nn_replica.py`)

- Same architecture in Keras
- SGD optimizer
- Same hyperparameters (learning rate, epochs, batch size)
- Used to validate NumPy implementation correctness
- Final accuracy: ~0.9268–0.9270

---

## Out-of-Box Challenge — Weight Visualization Experiment

`03_weight_visualization_experiment.py`

- Trains two models:
  - Real-label model
  - Shuffled-label model
- Same architecture, same data distribution
- Compares learned representations via first-layer weights

### Observations:

- Real-label model learns structured features (edges/strokes)
- Random-label model produces noisy, unstructured weights
- Clear difference between generalization vs memorization

Inspired by:

- Zhang et al. (2017): _Understanding Deep Learning Requires Rethinking Generalization_

---

## Results Summary

- NumPy model accuracy: ~0.8721
- Keras model accuracy: ~0.9268–0.9270
- Random-label model accuracy: ~0.10–0.15 (no learning beyond chance)
- Clear visual separation in learned features between models

---

## How to Run

```bash
python day-18/01_numpy_nn_scratch.py
python day-18/02_keras_nn_replica.py
python day-18/03_weight_visualization_experiment.py
```

---

## Data

- MNIST CSV (Kaggle Digit Recognizer format)
- Stored externally: `day-18/data/raw/train.csv`

---

## Outputs

- outputs/metrics → training logs
- outputs/models → saved models
- outputs/figures → weight visualizations
