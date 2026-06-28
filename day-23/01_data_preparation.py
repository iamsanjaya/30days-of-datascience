# %%
"""
01_data_preparation.py
Load the raw Twitter US Airline Sentiment CSV, apply both cleaning levels,
produce a stratified train/val/test split, and save everything to
data/processed/. Run this once before any of the model scripts (02-04),
and re-run it any time utils/data.py or utils/preprocessing.py change.
"""

import matplotlib.pyplot as plt

import config
from utils import data, preprocessing

# %%
print(f"Loading raw data from {config.RAW_CSV_PATH} ...")
raw_df = data.load_raw_tweets()
print(f"Raw rows: {len(raw_df)}")

df = data.clean_raw_frame(raw_df)
print(f"Rows after dropping nulls/duplicates: {len(df)}")

# %%
print("Applying light_clean (for BERT) and aggressive_clean (for TF-IDF / LSTM) ...")
df["light_clean"] = df[config.TEXT_COLUMN].apply(preprocessing.light_clean)
df["aggressive_clean"] = df[config.TEXT_COLUMN].apply(preprocessing.aggressive_clean)

# Drop rows that became empty after aggressive cleaning (e.g. tweets that were only a URL/@mention)
empty_after_clean = (df["aggressive_clean"].str.len() == 0).sum()
print(f"Dropping {empty_after_clean} rows that are empty after aggressive cleaning")
df = df[df["aggressive_clean"].str.len() > 0].reset_index(drop=True)


# %%
train_df, val_df, test_df = data.stratified_split(df)
print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

data.save_splits(
    train_df[
        [
            config.TEXT_COLUMN,
            "light_clean",
            "aggressive_clean",
            config.LABEL_COLUMN,
            "label_id",
        ]
    ],
    val_df[
        [
            config.TEXT_COLUMN,
            "light_clean",
            "aggressive_clean",
            config.LABEL_COLUMN,
            "label_id",
        ]
    ],
    test_df[
        [
            config.TEXT_COLUMN,
            "light_clean",
            "aggressive_clean",
            config.LABEL_COLUMN,
            "label_id",
        ]
    ],
)
print(f"Saved processed splits to {config.PROCESSED_DATA_DIR}")

# %%
print("\nClass distribution:")
for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
    print(f"  {name}: {data.class_distribution(split_df).to_dict()}")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
data.class_distribution(df).plot(
    kind="bar", ax=ax, color=["#c0392b", "#7f8c8d", "#27ae60"]
)
ax.set_title("Class Distribution — Full Cleaned Dataset")
ax.set_ylabel("Tweet count")
fig.tight_layout()
fig.savefig(config.OUTPUT_DIR / "class_distribution.png", dpi=config.FIGURE_DPI)
plt.close(fig)
print(f"Saved class_distribution.png to {config.OUTPUT_DIR}")
