# %%
"""
03_augmentation_experiment.py — Standard Task

For the smallest subset fractions (config.AUGMENTATION_TARGET_FRACTIONS),
retrain with data augmentation and compare against the no-augmentation result
already recorded by 02_train_subset_models.py for the same fraction.
"""

import json

import pandas as pd

import config
from utils.architecture import build_frozen_resnet_classifier
from utils.data import load_split_csv, make_dataset
from utils.training import evaluate_model, train_model

# %%
with open(config.DATA_EFFICIENCY_RESULTS_PATH) as f:
    no_aug_results = {r["fraction"]: r for r in json.load(f)}

val_df = load_split_csv(config.DAY21_VAL_CSV)
test_df = load_split_csv(config.DAY21_TEST_CSV)
val_ds = make_dataset(val_df, augment=False, shuffle=False)
test_ds = make_dataset(test_df, augment=False, shuffle=False)

# %%
comparisons: list[dict] = []

for fraction in config.AUGMENTATION_TARGET_FRACTIONS:
    if fraction not in no_aug_results:
        print(f"Skipping {fraction:.0%} — no baseline result found, run 02 first.")
        continue

    manifest_path = config.SUBSETS_DIR / f"subset_{int(fraction * 100):03d}pct.csv"
    subset_df = pd.read_csv(manifest_path)
    print(f"\n=== Re-training {fraction:.0%} subset WITH augmentation ===")

    train_ds = make_dataset(subset_df, augment=True, shuffle=True)
    model = build_frozen_resnet_classifier()
    train_model(
        model,
        train_ds,
        val_ds,
        epochs=config.EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_name=f"resnet50_subset_{int(fraction * 100):03d}pct_augmented",
        models_dir=config.MODELS_DIR,
    )

    metrics = evaluate_model(model, test_ds)
    no_aug_acc = no_aug_results[fraction]["test_accuracy"]
    print(
        f"No-aug accuracy: {no_aug_acc:.4f} | Aug accuracy: {metrics['accuracy']:.4f}"
    )

    comparisons.append(
        {
            "fraction": fraction,
            "no_aug_accuracy": no_aug_acc,
            "aug_accuracy": metrics["accuracy"],
        }
    )

# %%
with open(config.AUGMENTATION_RESULTS_PATH, "w") as f:
    json.dump(comparisons, f, indent=2)

print("\nSaved augmentation comparison to", config.AUGMENTATION_RESULTS_PATH)
