# %%
"""
02_train_subset_models.py — Standard Task

Train the same frozen-ResNet50 classifier on each subset fraction created by
01_create_subsets.py, evaluate every run on the SAME held-out Day 21 test
set, and record results so 04_compare_results.py can plot the degradation
curve.
"""

import json

import pandas as pd

import config
from utils.architecture import build_frozen_resnet_classifier
from utils.data import load_split_csv, make_dataset
from utils.training import evaluate_model, train_model

# %%
print("Loading fixed val/test sets (shared across all fraction experiments)...")
val_df = load_split_csv(config.DAY21_VAL_CSV)
test_df = load_split_csv(config.DAY21_TEST_CSV)
val_ds = make_dataset(val_df, augment=False, shuffle=False)
test_ds = make_dataset(test_df, augment=False, shuffle=False)
print(f"val={len(val_df)} images, test={len(test_df)} images")

# %%
results: list[dict] = []

for fraction in config.SUBSET_FRACTIONS:
    manifest_path = config.SUBSETS_DIR / f"subset_{int(fraction * 100):03d}pct.csv"
    subset_df = pd.read_csv(manifest_path)
    print(f"\n=== Training on {fraction:.0%} subset ({len(subset_df)} images) ===")

    train_ds = make_dataset(subset_df, augment=False, shuffle=True)
    model = build_frozen_resnet_classifier()
    train_model(
        model,
        train_ds,
        val_ds,
        epochs=config.EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_name=f"resnet50_subset_{int(fraction * 100):03d}pct",
        models_dir=config.MODELS_DIR,
    )

    metrics = evaluate_model(model, test_ds)
    print(f"Test accuracy at {fraction:.0%}: {metrics['accuracy']:.4f}")

    results.append(
        {
            "fraction": fraction,
            "train_size": len(subset_df),
            "test_accuracy": metrics["accuracy"],
            "test_loss": metrics["loss"],
        }
    )

# %%
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(config.DATA_EFFICIENCY_RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\nAll fractions trained. Results saved to", config.DATA_EFFICIENCY_RESULTS_PATH)
for r in results:
    print(
        f"  {r['fraction']:.0%}: n={r['train_size']:>5} -> acc={r['test_accuracy']:.4f}"
    )
