# %%
"""Day 20 — 01: Verify the CIFAR-10 download and inspect class balance.

Run this first. It only reads data — no training happens here.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

import config
from utils import data

# %% Load both splits and confirm shapes
train_images, train_labels = data.load_split(config.TRAIN_DIR)
test_images, test_labels = data.load_split(config.TEST_DIR)

print(f"Train: {train_images.shape}, labels: {train_labels.shape}")
print(f"Test:  {test_images.shape}, labels: {test_labels.shape}")

# %% Per-class counts
train_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
test_counts = np.bincount(test_labels, minlength=config.NUM_CLASSES)

for name, train_c, test_c in zip(config.CLASS_NAMES, train_counts, test_counts):
    print(f"{name:>12}: train={train_c:>5}  test={test_c:>5}")

# %% Class balance chart
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(config.NUM_CLASSES)
ax.bar(x - 0.2, train_counts, width=0.4, label="train")
ax.bar(x + 0.2, test_counts, width=0.4, label="test")
ax.set_xticks(x)
ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
ax.set_ylabel("image count")
ax.set_title("CIFAR-10 class balance — train vs test")
ax.legend()
fig.tight_layout()
fig.savefig(config.OUTPUTS_DIR / "class_balance.png", dpi=300)
plt.close(fig)

# %% Confirm the overfit subset is small and reproducible
x_train, y_train, x_val, y_val = data.get_overfit_subset()
print(f"\nOverfit subset — train: {x_train.shape}, val: {x_val.shape}")
print("This subset is deterministic: rerunning this cell always returns the")
print("identical 500 train / 200 val images (fixed seed in config.py).")
