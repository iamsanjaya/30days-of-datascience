# %%
"""
01_create_subsets.py — Standard Task

Create stratified, class-balanced subsets of the Day 21 Cats vs Dogs training
set at 100%, 50%, 25%, 10%, and 5%. Saves a manifest CSV per fraction so that
every later script trains on an exactly reproducible subset.
"""

from pathlib import Path

import config
from utils.data import load_split_csv, stratified_subset

# %%
print("Loading Day 21 train split manifest...")
train_df = load_split_csv(config.DAY21_TRAIN_CSV)
print(
    f"Full train set: {len(train_df)} images "
    f"({(train_df['label'] == 0).sum()} cats, {(train_df['label'] == 1).sum()} dogs)"
)

# %%
config.SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

for fraction in config.SUBSET_FRACTIONS:
    subset_df = stratified_subset(train_df, fraction, seed=config.SEED)
    out_path: Path = config.SUBSETS_DIR / f"subset_{int(fraction * 100):03d}pct.csv"
    subset_df.to_csv(out_path, index=False)
    n_cat = (subset_df["label"] == 0).sum()
    n_dog = (subset_df["label"] == 1).sum()
    print(
        f"fraction={fraction:.2f} -> {len(subset_df)} images "
        f"({n_cat} cats, {n_dog} dogs) -> saved {out_path.name}"
    )

print("Done. Manifests saved to", config.SUBSETS_DIR)
