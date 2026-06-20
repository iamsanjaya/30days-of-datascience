# Day 18 — Neural Networks from Scratch + Weight Visualization Experiment

**Date:** June 20, 2026
**Dataset:** MNIST Digit Recognizer (train.csv) (https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer)

## What I built today:

- Implemented a 2-layer neural network from scratch using NumPy
- Built a Keras replica to validate correctness of manual backprop implementation
- Created a shuffled-label experiment to test memorization vs learning
- Added weight visualization for interpretability of learned features
- Built a full reproducible training + evaluation pipeline

---

## The out-of-box challenge result:

- NumPy model achieved ~0.8721 accuracy on MNIST
- Keras replica achieved ~0.9268–0.9270 accuracy
- Random-label model stayed near ~0.10–0.15 accuracy (no generalization)
- Weight visualization showed:
  - structured patterns for real-label training
  - noise-like patterns for shuffled-label training

---

## What surprised me:

- Keras outperformed NumPy despite identical architecture
- Random-label training still slightly improves above pure chance
- Visual differences in learned weights are very clear and interpretable
- Small implementation differences significantly affect convergence

---

## What I don't fully understand yet:

- Why random-label training consistently rises above exact 0.10 baseline
- Why Keras converges faster than NumPy under same settings
- Sensitivity of weight visualization to initialization and training randomness
- Contribution of optimizer implementation differences vs architecture

---

## GitHub commit made: ✅

`Day 18 — Neural Networks from Scratch + Weight Visualization Experiment`

---

## Tomorrow's priority:

Day 19 — Visualization + reproducibility improvements. Make plotting reusable across models, add real vs random comparison charts, and strengthen experiment tracking for fully reproducible runs (params, metrics, splits).
