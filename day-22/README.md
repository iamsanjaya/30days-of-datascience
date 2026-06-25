# Day 22 — CNN: Data Efficiency and Niche Domain Classifier

## Problem Statement

How much training data does a CNN actually need, and what can you do when
you don't have much of it? Two related questions, two experiments:

1. **Standard task:** How does test accuracy degrade as the Cats vs Dogs
   training set shrinks from 100% down to 5%? Does augmentation help recover
   some of that lost performance at the smallest sizes?
2. **Out-of-box challenge:** Starting from a deliberately tiny, real-world-
   realistic pool of **450 images** (under the 500-image cap) of Nepali
   Devanagari handwritten digits, how high can accuracy be pushed using every
   data-efficiency trick available — augmentation, transfer learning,
   test-time augmentation, and pseudo-labeling?

---

## Datasets

| Task       | Dataset                                                          | Source                                                                                  |
| ---------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Standard   | Microsoft Cats vs Dogs (cleaned splits from Day 21)              | Reused from Day 21                                                                      |
| Out-of-box | Devanagari Handwritten Character Dataset — 10 digit classes only | https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase |

No synthetic data is used anywhere. The "unlabeled" pool used for pseudo-labeling is composed of real images whose labels were intentionally hidden during training. True labels were retained only for evaluation of pseudo-label quality.

---

## Approach

### Standard Task (Scripts 01–04)

A fixed transfer-learning architecture was used throughout the experiment so the only changing variable was training-set size.

- Created stratified subsets at 100%, 50%, 25%, 10%, and 5%.
- Trained identical ResNet50-based classifiers on each subset.
- Evaluated all models on the same validation and test sets.
- Re-trained the smallest subsets with augmentation.
- Generated comparison plots and data-efficiency curves.

### Out-of-Box Challenge (Scripts 05–11)

A realistic low-data computer vision workflow was implemented.

- Constructed a 450-image training pool.
- Trained a baseline CNN from scratch.
- Added augmentation.
- Applied MobileNetV2 transfer learning.
- Fine-tuned the top backbone layers.
- Evaluated with Test-Time Augmentation (TTA).
- Performed pseudo-labeling using a separate unlabeled pool.
- Compared all techniques.

---

## Project Structure

```
day-22-cnn-data-efficiency/
├── config.py                         # Central configuration and dataset paths
├── 01_create_subsets.py              # Creates stratified Cats vs Dogs subsets (100%→5%)
├── 02_train_models.py                # Trains ResNet50 models on each subset
├── 03_augmentation_experiment.py     # Tests augmentation on small-data subsets
├── 04_compare_results.py             # Generates accuracy curves and comparison plots
├── 05_prepare_niche_dataset.py       # Creates train/val/test/unlabeled digit splits
├── 06_train_baseline_cnn.py          # Trains a CNN from scratch on 450 images
├── 07_train_with_augmentation.py     # Trains CNN with data augmentation
├── 08_train_transfer_learning.py     # MobileNetV2 transfer learning and fine-tuning
├── 09_evaluate_with_tta.py           # Evaluates model using Test-Time Augmentation
├── 10_pseudo_labeling.py             # Semi-supervised learning with pseudo-labels
├── 11_compare_niche_results.py       # Compares all niche-domain techniques
├── utils/
│   ├── data.py                       # Dataset loading and subset generation utilities
│   ├── architecture.py               # ResNet50 classifier architecture
│   ├── training.py                   # Shared training and evaluation functions
│   ├── visualization.py              # Plotting and result visualization helpers
│   ├── niche_data.py                 # Devanagari dataset and TTA utilities
│   └── niche_architecture.py         # Baseline CNN and MobileNetV2 architectures
├── outputs/                          # Generated manifests, JSON results, and plots
│   ├── niche_manifests/              # Dataset split manifests
│   │   ├── test.csv                  # Test set image paths and labels
│   │   ├── train.csv                 # 450-image training pool
│   │   ├── unlabeled.csv             # Unlabeled pool for pseudo-labeling
│   │   └── val.csv                   # Validation set image paths and labels
│   ├── plots/                        # Generated visualizations
│   │   ├── augmentation_comparison.png      # Augmentation vs no-augmentation comparison
│   │   ├── data_efficiency_curve.png        # Accuracy vs training-data-size curve
│   │   └── niche_techniques_comparison.png  # Niche-domain technique comparison chart
│   ├── results/                             # Experiment metrics and summaries
│   │   ├── augmentation_results.json        # Augmentation experiment results
│   │   ├── data_efficiency_results.json     # Data-efficiency experiment metrics
│   │   ├── niche_results.json              # Niche-domain experiment results
│   │   └── niche_summary_table.csv         # Final technique comparison table
│   └── subsets/                      # Stratified Cats vs Dogs subsets
│       ├── subset_005pct.csv         # 5% training subset manifest
│       ├── subset_010pct.csv         # 10% training subset manifest
│       ├── subset_025pct.csv         # 25% training subset manifest
│       ├── subset_050pct.csv         # 50% training subset manifest
│       └── subset_100pct.csv         # Full training set manifest
├── models/                           # Saved trained model checkpoints
│   ├── niche_baseline_cnn.keras              # Baseline CNN trained from scratch
│   ├── niche_cnn_augmented.keras             # CNN trained with augmentation
│   ├── niche_pseudo_labeled_finetuned.keras  # Fine-tuned transfer model after pseudo-labeling
│   ├── niche_pseudo_labeled_frozen.keras     # Frozen-backbone model after pseudo-labeling
│   ├── niche_transfer_finetuned.keras        # Fine-tuned MobileNetV2 transfer model
│   ├── niche_transfer_frozen.keras           # Frozen-backbone MobileNetV2 model
│   ├── niche_transfer_model.keras            # Best transfer-learning model used for evaluation
│   ├── resnet50_subset_005pct.keras          # ResNet50 trained on 5% Cats vs Dogs data
│   ├── resnet50_subset_005pct_augmented.keras # 5% subset model with augmentation
│   ├── resnet50_subset_010pct.keras          # ResNet50 trained on 10% data
│   ├── resnet50_subset_010pct_augmented.keras # 10% subset model with augmentation
│   ├── resnet50_subset_025pct.keras          # ResNet50 trained on 25% data
│   ├── resnet50_subset_050pct.keras          # ResNet50 trained on 50% data
│   └── resnet50_subset_100pct.keras          # ResNet50 trained on full dataset
└── README.md                         # Project documentation and experiment analysis

learning-journal/
    └── day-22.md
```

---

## How to Run

```bash
# Standard task
python 01_create_subsets.py
python 02_train_models.py
python 03_augmentation_experiment.py
python 04_compare_results.py

# Niche-domain challenge
python 05_prepare_niche_dataset.py
python 06_train_baseline_cnn.py
python 07_train_with_augmentation.py
python 08_train_transfer_learning.py
python 09_evaluate_with_tta.py
python 10_pseudo_labeling.py
python 11_compare_niche_results.py
```

---

## Results

### Experiment 1 — CNN Data Efficiency (Cats vs Dogs)

#### Dataset Sizes

| Fraction | Images |
| -------- | -----: |
| 100%     | 17,493 |
| 50%      |  8,746 |
| 25%      |  4,373 |
| 10%      |  1,749 |
| 5%       |    874 |

#### Test Accuracy

| Training Data | Test Accuracy |
| ------------- | ------------: |
| 100%          |        98.72% |
| 50%           |        98.67% |
| 25%           |        98.85% |
| 10%           |        98.40% |
| 5%            |        98.48% |

#### Augmentation Comparison

| Training Fraction | No Augmentation | With Augmentation |
| ----------------- | --------------: | ----------------: |
| 10%               |          98.40% |            98.75% |
| 5%                |          98.48% |            98.40% |

### Findings

- Performance remained extremely stable despite removing 95% of the training data.
- The 25% subset slightly outperformed the full dataset.
- Transfer learning dramatically reduced data requirements.
- Augmentation provided only marginal benefit because performance was already near saturation.

---

### Experiment 2 — Niche Domain Classifier (Nepali Devanagari Digits)

#### Dataset Split

| Split          | Images |
| -------------- | -----: |
| Train          |    450 |
| Validation     |    150 |
| Test           |    150 |
| Unlabeled Pool |  1,500 |

#### Technique Comparison

| Technique                              | Test Accuracy |
| -------------------------------------- | ------------: |
| Baseline CNN                           |        12.00% |
| CNN + Augmentation                     |        10.67% |
| Transfer Learning (Frozen MobileNetV2) |        90.67% |
| Transfer Learning + Fine-Tuning        |        90.67% |
| Transfer Learning + TTA                |        91.33% |
| Transfer Learning + Pseudo-Labeling    |        91.33% |

#### Pseudo-Labeling Statistics

| Metric                 |  Value |
| ---------------------- | -----: |
| Unlabeled Images       |  1,500 |
| Accepted Pseudo Labels |    893 |
| Confidence Threshold   |    90% |
| Pseudo-Label Accuracy  | 99.33% |
| Expanded Training Pool |  1,343 |

### Findings

- CNN training from scratch failed due to limited labeled data.
- Augmentation alone was insufficient.
- Transfer learning increased accuracy from 12.00% to 90.67%.
- Fine-tuning produced no measurable improvement.
- TTA improved inference accuracy to 91.33%.
- Pseudo-labeling nearly tripled the effective training set size and matched the best TTA result.

---

## Strategy Document (Out-of-Box Deliverable)

### What Worked

#### Transfer Learning

The largest gain came from transfer learning.

Accuracy improved from:

```text
12.00% → 90.67%
```

The pretrained MobileNetV2 backbone already contained useful visual representations, allowing the model to generalize despite having only 450 labeled images.

#### Test-Time Augmentation

TTA increased performance from:

```text
90.67% → 91.33%
```

by averaging predictions across multiple augmented versions of the same image.

#### Pseudo-Labeling

Using a 90% confidence threshold, 893 additional images were automatically labeled and added to the training set.

This expanded the effective training set from:

```text
450 → 1343 images
```

while maintaining 99.33% pseudo-label accuracy.

---

### What Did Not Help

#### Training From Scratch

The baseline CNN achieved only 12.00% accuracy because there were too few labeled samples to learn robust features.

#### Augmentation Alone

Data augmentation increased visual diversity but could not compensate for weak learned representations.

#### Fine-Tuning

Fine-tuning the upper MobileNetV2 layers produced no measurable improvement over the frozen backbone model.

---

### What I Would Do With A Real 450-Image Client Dataset

1. Start with transfer learning immediately.
2. Collect additional unlabeled examples whenever possible.
3. Use pseudo-labeling to expand the dataset.
4. Apply stronger augmentation policies.
5. Use k-fold cross-validation for more reliable evaluation.
6. Experiment with EfficientNet and ConvNeXt backbones.
7. Introduce active learning to prioritize annotation effort.
8. Explore self-supervised pretraining if large unlabeled datasets are available.

---

## Key Lessons Learned

### Data Efficiency

Modern pretrained CNNs can maintain excellent performance even after dramatic reductions in training data.

### Transfer Learning

Transfer learning remains the most effective strategy for small-data computer vision problems.

### Augmentation

Augmentation is most effective when paired with strong pretrained features rather than used as a standalone solution.

### Semi-Supervised Learning

Pseudo-labeling can successfully leverage unlabeled data when confidence filtering is used carefully.

### Fine-Tuning

Fine-tuning is not always beneficial. For extremely small datasets, a frozen backbone may already be near optimal.

---

## Final Conclusion

This project explored two complementary aspects of CNN data efficiency.

The Cats vs Dogs experiment demonstrated that pretrained CNNs can maintain nearly identical performance even after losing 95% of their training data.

The Nepali Devanagari digit experiment demonstrated that transfer learning, test-time augmentation, and pseudo-labeling can transform a small 450-image dataset into a highly effective classifier achieving 91.33% test accuracy.

The strongest overall lesson from Day 22 is that modern computer vision systems derive much of their power from pretrained representations, allowing them to perform well even when labeled data is scarce.
