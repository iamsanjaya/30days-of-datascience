"""
Day 21 — Step 1: Data Preparation

Scans data/raw/PetImages/{Cat,Dog}/ (Microsoft Cats vs Dogs dataset),
validates every file (drops zero-byte / undecodable images — this
dataset is known to ship with a few, e.g. Cat/666.jpg, Dog/11702.jpg),
builds a stratified train/val/test split, and saves the split
manifests as CSVs so every downstream script works from the same
fixed, already-cleaned split.

Run:
    python 01_prepare_data.py
"""

import config
from utils import data


def main() -> None:
    print(f"Scanning raw data directory: {config.RAW_DATA_DIR}")
    df = data.scan_raw_directory()
    print(
        f"Found {len(df)} images on disk ({(df['label'] == 0).sum()} cats, "
        f"{(df['label'] == 1).sum()} dogs)"
    )

    print("Validating every file (this reads each image once — takes a few minutes)...")
    clean_df, removed_df = data.validate_images(df)

    if len(removed_df) > 0:
        print(f"Removed {len(removed_df)} corrupt/zero-byte files:")
        for _, row in removed_df.iterrows():
            print(f"  {row['filepath']} -> {row['reason']}")
        data.save_corrupt_files_log(removed_df)
        print(f"Full list saved to {config.CORRUPT_FILES_LOG}")
    else:
        print("No corrupt files found.")

    print(
        f"Clean dataset: {len(clean_df)} images "
        f"({(clean_df['label'] == 0).sum()} cats, {(clean_df['label'] == 1).sum()} dogs)"
    )

    train_df, val_df, test_df = data.make_splits(clean_df)
    print(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cat_count = (split_df["label"] == 0).sum()
        dog_count = (split_df["label"] == 1).sum()
        print(f"  {name}: {cat_count} cats / {dog_count} dogs")

    data.save_splits(train_df, val_df, test_df)
    print(f"Splits saved to {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()
