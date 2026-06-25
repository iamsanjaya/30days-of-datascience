# %%
"""
07_train_with_augmentation.py — Out-of-Box Challenge: augmentation trick

Same tiny CNN architecture as 06, same 450-image pool — the only change is
augment=True on the training dataset. Isolates the effect of augmentation
from every other trick.
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

train_ds = make_niche_dataset(train_df, augment=True, shuffle=True)
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
    model_name="niche_cnn_augmented",
    models_dir=config.MODELS_DIR,
)
metrics = evaluate_model(model, test_ds)
print(f"Augmented CNN test accuracy: {metrics['accuracy']:.4f}")

# %%
with open(config.NICHE_RESULTS_PATH) as f:
    all_results = json.load(f)

all_results["cnn_augmented"] = {
    "accuracy": metrics["accuracy"],
    "loss": metrics["loss"],
}
with open(config.NICHE_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved result to", config.NICHE_RESULTS_PATH)
