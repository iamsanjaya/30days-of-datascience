# %%
"""
08_train_transfer_learning.py — Out-of-Box Challenge: transfer learning trick

Frozen MobileNetV2 backbone + trainable head, trained on the augmented
450-image pool, then a short fine-tune pass with the top backbone layers
unfrozen at a low learning rate. Saves the final model — 09 (TTA) and
10 (pseudo-labeling) both build on top of it.
"""

import json

import pandas as pd

import config
from utils.niche_architecture import build_transfer_model, unfreeze_for_fine_tuning
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
print("=== Stage 1: train frozen-backbone head ===")
model = build_transfer_model()
train_model(
    model,
    train_ds,
    val_ds,
    epochs=config.NICHE_EPOCHS,
    patience=config.NICHE_EARLY_STOPPING_PATIENCE,
    model_name="niche_transfer_frozen",
    models_dir=config.MODELS_DIR,
)
frozen_metrics = evaluate_model(model, test_ds)
print(f"Frozen-backbone test accuracy: {frozen_metrics['accuracy']:.4f}")

# %%
print("\n=== Stage 2: fine-tune top backbone layers ===")
model = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=20)
train_model(
    model,
    train_ds,
    val_ds,
    epochs=config.NICHE_FINE_TUNE_EPOCHS,
    patience=config.NICHE_EARLY_STOPPING_PATIENCE,
    model_name="niche_transfer_finetuned",
    models_dir=config.MODELS_DIR,
)
finetuned_metrics = evaluate_model(model, test_ds)
print(f"Fine-tuned test accuracy: {finetuned_metrics['accuracy']:.4f}")

# %%
model.save(config.MODELS_DIR / "niche_transfer_model.keras")

with open(config.NICHE_RESULTS_PATH) as f:
    all_results = json.load(f)

all_results["transfer_learning_frozen"] = {
    "accuracy": frozen_metrics["accuracy"],
    "loss": frozen_metrics["loss"],
}
all_results["transfer_learning"] = {
    "accuracy": finetuned_metrics["accuracy"],
    "loss": finetuned_metrics["loss"],
}
with open(config.NICHE_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved results to", config.NICHE_RESULTS_PATH)
print("Saved model to", config.MODELS_DIR / "niche_transfer_model.keras")
