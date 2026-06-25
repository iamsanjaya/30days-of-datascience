# %%
"""
06_train_baseline_cnn.py — Out-of-Box Challenge: "no tricks" baseline

Train a small CNN from scratch on the raw 450-image training pool, with no
augmentation and no transfer learning. This is the number every later trick
gets compared against.
"""

import json

import pandas as pd

import config
from utils.niche_architecture import build_tiny_cnn
from utils.niche_data import make_niche_dataset
from utils.training import evaluate_model, train_model

# %%
train_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "train.csv")
val_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "val.csv")
test_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "test.csv")
print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

train_ds = make_niche_dataset(train_df, augment=False, shuffle=True)
val_ds = make_niche_dataset(val_df, augment=False, shuffle=False)
test_ds = make_niche_dataset(test_df, augment=False, shuffle=False)

# %%
model = build_tiny_cnn()
train_model(
    model,
    train_ds,
    val_ds,
    epochs=config.NICHE_EPOCHS,
    patience=config.NICHE_EARLY_STOPPING_PATIENCE,
    model_name="niche_baseline_cnn",
    models_dir=config.MODELS_DIR,
)
metrics = evaluate_model(model, test_ds)
print(f"Baseline CNN test accuracy: {metrics['accuracy']:.4f}")

# %%
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
try:
    with open(config.NICHE_RESULTS_PATH) as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = {}

all_results["baseline_cnn"] = {"accuracy": metrics["accuracy"], "loss": metrics["loss"]}
with open(config.NICHE_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved result to", config.NICHE_RESULTS_PATH)
