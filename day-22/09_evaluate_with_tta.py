# %%
"""
09_evaluate_with_tta.py — Out-of-Box Challenge: test-time augmentation (TTA)

Load the fine-tuned transfer model from 08 and re-evaluate the test set using
TTA: average predictions over several augmented copies of each image instead
of a single forward pass.
"""

import json

import pandas as pd
import tensorflow as tf

import config
from utils.niche_data import predict_with_tta

# %%
model = tf.keras.models.load_model(config.MODELS_DIR / "niche_transfer_model.keras")
test_df = pd.read_csv(config.NICHE_MANIFESTS_DIR / "test.csv")

# %%
accuracy, _ = predict_with_tta(
    model,
    test_df,
    num_augmentations=config.TTA_NUM_AUGMENTATIONS,
    seed=config.SEED,
)
print(
    f"TTA test accuracy ({config.TTA_NUM_AUGMENTATIONS} augmentations/image): {accuracy:.4f}"
)

# %%
with open(config.NICHE_RESULTS_PATH) as f:
    all_results = json.load(f)

all_results["transfer_learning_tta"] = {"accuracy": accuracy}
with open(config.NICHE_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved result to", config.NICHE_RESULTS_PATH)

no_tta_acc = all_results.get("transfer_learning", {}).get("accuracy")
if no_tta_acc is not None:
    delta = accuracy - no_tta_acc
    print(f"TTA vs single-pass: {delta:+.4f}")
