# %%
"""
10_pseudo_labeling.py — Out-of-Box Challenge: pseudo-labeling trick

Use the fine-tuned transfer model from 08 to predict labels on the
"unlabeled" pool (real images, true labels withheld during this step).
Keep only predictions above config.PSEUDO_LABEL_CONFIDENCE_THRESHOLD, merge
them with the original 450-image training pool, and retrain from scratch.

True labels for the unlabeled pool are loaded separately ONLY to report how
accurate the pseudo-labels actually were — never used during training. This
is the honest way to quantify a real risk of this technique: confidently
wrong pseudo-labels reinforcing the model's own mistakes.
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from utils.niche_architecture import build_transfer_model, unfreeze_for_fine_tuning
from utils.niche_data import make_niche_dataset
from utils.training import evaluate_model, train_model

# %%
model = tf.keras.models.load_model(config.MODELS_DIR / "niche_transfer_model.keras")

train_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "train.csv")
val_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "val.csv")
test_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "test.csv")
unlabeled_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "unlabeled.csv")
true_unlabeled_labels = unlabeled_df["label"].to_numpy()  # held back for reporting only

# %%
print(f"Predicting on {len(unlabeled_df)} 'unlabeled' images...")
unlabeled_ds = make_niche_dataset(unlabeled_df, augment=False, shuffle=False)
probs = model.predict(unlabeled_ds, verbose=0)
confidences = np.max(probs, axis=1)
predicted_labels = np.argmax(probs, axis=1)

keep_mask = confidences >= config.PSEUDO_LABEL_CONFIDENCE_THRESHOLD
n_kept = int(keep_mask.sum())
pseudo_label_accuracy = (
    float(np.mean(predicted_labels[keep_mask] == true_unlabeled_labels[keep_mask]))
    if n_kept
    else 0.0
)

print(
    f"Kept {n_kept}/{len(unlabeled_df)} predictions above "
    f"{config.PSEUDO_LABEL_CONFIDENCE_THRESHOLD:.0%} confidence"
)
print(
    f"Pseudo-label accuracy among kept examples (vs true labels, for reporting only): "
    f"{pseudo_label_accuracy:.4f}"
)

# %%
pseudo_df = unlabeled_df.loc[keep_mask, ["filepath"]].copy()
pseudo_df["label"] = predicted_labels[keep_mask]
combined_train_df = pd.concat([train_df, pseudo_df]).reset_index(drop=True)
print(
    f"Combined training pool: {len(train_df)} original + {len(pseudo_df)} pseudo-labeled "
    f"= {len(combined_train_df)} images"
)

# %%
train_ds = make_niche_dataset(combined_train_df, augment=True, shuffle=True)
val_ds = make_niche_dataset(val_df, augment=False, shuffle=False)
test_ds = make_niche_dataset(test_df, augment=False, shuffle=False)

model = build_transfer_model()
train_model(
    model,
    train_ds,
    val_ds,
    epochs=config.NICHE_EPOCHS,
    patience=config.NICHE_EARLY_STOPPING_PATIENCE,
    model_name="niche_pseudo_labeled_frozen",
    models_dir=config.MODELS_DIR,
)
model = unfreeze_for_fine_tuning(model, num_layers_to_unfreeze=20)
train_model(
    model,
    train_ds,
    val_ds,
    epochs=config.NICHE_FINE_TUNE_EPOCHS,
    patience=config.NICHE_EARLY_STOPPING_PATIENCE,
    model_name="niche_pseudo_labeled_finetuned",
    models_dir=config.MODELS_DIR,
)

metrics = evaluate_model(model, test_ds)
print(f"\nPseudo-labeled-retrained test accuracy: {metrics['accuracy']:.4f}")

# %%
with open(config.NICHE_RESULTS_PATH) as f:
    all_results = json.load(f)

all_results["pseudo_labeling"] = {
    "accuracy": metrics["accuracy"],
    "loss": metrics["loss"],
    "n_pseudo_labels_kept": n_kept,
    "pseudo_label_accuracy": pseudo_label_accuracy,
}
with open(config.NICHE_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved result to", config.NICHE_RESULTS_PATH)
