# Day 22 — June 24, 2026

## What I built today:

Built two computer vision experiments focused on CNN data efficiency and low-data image classification.

### Experiment 1 — Data Efficiency Study (Cats vs Dogs)

- Created stratified training subsets at 100%, 50%, 25%, 10%, and 5% of the original training set.
- Trained the same ResNet50-based transfer learning model on each subset.
- Evaluated how performance changed as training data was reduced.
- Tested whether augmentation improved results on the smallest datasets.
- Generated comparison plots and data-efficiency curves.

Results:

| Training Fraction | Test Accuracy |
| ----------------- | ------------: |
| 100%              |        98.72% |
| 50%               |        98.67% |
| 25%               |        98.85% |
| 10%               |        98.40% |
| 5%                |        98.48% |

### Experiment 2 — Niche Domain Classifier (Nepali Devanagari Digits)

- Created a small-data classification problem using only 450 labeled images.
- Built a baseline CNN from scratch.
- Tested augmentation.
- Applied transfer learning with MobileNetV2.
- Fine-tuned the pretrained backbone.
- Evaluated test-time augmentation (TTA).
- Implemented pseudo-labeling using a separate unlabeled pool.
- Compared all approaches in a final leaderboard.

Final Results:

| Technique                           | Test Accuracy |
| ----------------------------------- | ------------: |
| Baseline CNN                        |        12.00% |
| CNN + Augmentation                  |        10.67% |
| Transfer Learning                   |        90.67% |
| Transfer Learning + Fine-Tuning     |        90.67% |
| Transfer Learning + TTA             |        91.33% |
| Transfer Learning + Pseudo-Labeling |        91.33% |

---

## The out-of-box challenge result:

Using only 450 labeled images of Nepali Devanagari handwritten digits, I improved accuracy from 12.00% with a CNN trained from scratch to 91.33% using transfer learning and data-efficiency techniques.

Pseudo-labeling expanded the effective training set from 450 images to 1,343 images by automatically labeling 893 high-confidence samples from a 1,500-image unlabeled pool.

Best result:

**91.33% test accuracy**

Techniques responsible for the improvement:

- Transfer Learning
- Test-Time Augmentation (TTA)
- Pseudo-Labeling

---

## What surprised me:

- Reducing the Cats vs Dogs dataset by 95% barely affected accuracy.
- The 25% subset slightly outperformed the full dataset.
- Data augmentation alone did not improve the niche-domain classifier.
- Transfer learning produced an enormous jump from 12.00% to 90.67%.
- Fine-tuning the backbone provided no improvement despite being a commonly recommended step.
- The pseudo-labels were extremely accurate (99.33%), making pseudo-labeling much more effective than expected.

---

## What I don't fully understand yet:

- Why the 25% Cats vs Dogs subset slightly outperformed the full training set.
- Why fine-tuning MobileNetV2 produced no measurable gain over the frozen backbone.
- How self-supervised learning compares against pseudo-labeling on very small datasets.
- When test-time augmentation is worth the additional inference cost in production systems.
- How active learning could further reduce the amount of labeled data required.

---

## GitHub commit made: ✅

`Day 22 — CNN: Data Efficiency and Niche Domain Classifier `

---

## Tomorrow's priority:

Day 23 — NLP Fundamentals and BoW to BERT + Failure Forensics

Move from Bag-of-Words and TF-IDF to modern transformer-based NLP models, including BERT, while developing a deeper understanding of text representation, semantic similarity, and systematic NLP failure analysis.
