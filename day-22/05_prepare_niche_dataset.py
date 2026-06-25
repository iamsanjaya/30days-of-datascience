# %%
"""
05_prepare_niche_dataset.py — Out-of-Box Challenge

From the full Kaggle "Devanagari Handwritten Character Dataset", carve out:
- train pool:      45/class x 10 classes = 450 images  (the deliberate constraint)
- val pool:         15/class  (model selection during training)
- test pool:        15/class  (final, untouched evaluation)
- unlabeled pool:  150/class  (real images, labels withheld — used only by
                                10_pseudo_labeling.py to simulate having
                                unlabeled data available)

Train + Test from the original dataset are merged first so the carve-out
draws from the largest possible pool per class (the dataset has ~2000
images/class natively, far more than we need).
"""

import pandas as pd

import config
from utils.niche_data import sample_non_overlapping_pools, scan_digit_classes

# %%
print("Scanning full Devanagari digit dataset...")
train_split = scan_digit_classes(config.NICHE_RAW_DATA_DIR / config.NICHE_TRAIN_SUBDIR)
test_split = scan_digit_classes(config.NICHE_RAW_DATA_DIR / config.NICHE_TEST_SUBDIR)
full_df = pd.concat([train_split, test_split]).reset_index(drop=True)
print(
    f"Full pool: {len(full_df)} images across {full_df['label'].nunique()} digit classes"
)
print(full_df.groupby("label").size())

# %%
pools = sample_non_overlapping_pools(
    full_df,
    train_per_class=config.NICHE_TRAIN_PER_CLASS,
    val_per_class=config.NICHE_VAL_PER_CLASS,
    test_per_class=config.NICHE_TEST_PER_CLASS,
    unlabeled_per_class=config.NICHE_UNLABELED_POOL_PER_CLASS,
    seed=config.SEED,
)

config.NICHE_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
for name, df in pools.items():
    out_path = config.NICHE_MANIFESTS_DIR / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"{name}: {len(df)} images -> {out_path.name}")

assert (
    len(pools["train"]) < 500
), "Train pool must stay under 500 images — that's the challenge constraint."
print(
    "\nTrain pool confirmed under 500 images. Manifests saved to",
    config.NICHE_MANIFESTS_DIR,
)
