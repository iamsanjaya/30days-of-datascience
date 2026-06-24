# Day 21 — June 23, 2026

## What I built today:

Built a complete transfer learning image classification pipeline using TensorFlow and Keras on the Microsoft Cats vs Dogs dataset.

The project included:

- Full dataset scanning and validation
- Automatic corrupt image detection and removal
- Stratified train/validation/test splitting
- tf.data input pipeline with parallel loading and prefetching
- Transfer learning using ImageNet-pretrained CNN backbones
- Frozen feature extraction training phase
- Fine-tuning phase with selective layer unfreezing
- Early stopping and learning rate scheduling
- Model checkpointing
- Grad-CAM explainability analysis
- Multi-backbone experimentation (ResNet50 and Xception)
- Automated backbone performance comparison

---

## The out-of-box challenge result:

Implemented Grad-CAM visualizations for correctly classified and misclassified test samples.

The heatmaps showed that the model generally focused on meaningful image regions such as:

- Animal faces
- Eyes
- Fur texture
- Ears and snouts

Most misclassifications occurred on visually ambiguous samples rather than because the model was focusing on completely irrelevant background regions.

This demonstrated that the learned features were largely meaningful and that remaining errors were primarily due to difficult examples near the decision boundary.

---

## What surprised me:

Several findings stood out:

- Only 9 images out of nearly 25,000 were corrupted and required removal.
- ResNet50 achieved over 99% test accuracy with relatively little fine-tuning.
- Fine-tuning improved performance only marginally because pretrained ImageNet features were already extremely effective.
- Xception consistently outperformed ResNet50 despite using the exact same dataset split.
- A small architecture change produced measurable gains even at performance levels above 99%.

---

## What I don't fully understand yet:

Topics I want to explore further:

- Why Xception generalizes better than ResNet50 on certain image classification tasks.
- Internal feature representations learned during transfer learning.
- Advanced Grad-CAM variants and explainability methods.
- When to fine-tune deeper portions of a pretrained network versus only the final layers.
- Trade-offs between CNN architectures and Vision Transformers.

---

## Final Results

### ResNet50 (Fine-Tuned)

- Accuracy: 99.01%
- Precision: 99.30%
- Recall: 98.72%
- F1 Score: 99.01%
- ROC-AUC: 99.95%

### Xception (Fine-Tuned)

- Accuracy: 99.47%
- Precision: 99.52%
- Recall: 99.41%
- F1 Score: 99.47%
- ROC-AUC: 99.96%

### Best Backbone

Xception achieved the best overall performance and became the final selected model.

---

## GitHub commit made: ✅

`Day 21 — CNNs: Transfer Learning (ResNet50 / Xception) + Grad-CAM`

## Tomorrow's priority:

Day 22 — Data Efficiency and Niche Domain Classifier.

Move deeper into practical computer vision by exploring data-efficient learning, stronger augmentation strategies, domain-specific image classification, and advanced transfer learning workflows while continuing to strengthen deep learning engineering fundamentals.
