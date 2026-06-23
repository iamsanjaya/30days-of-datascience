# %%
"""Day 20 — 03: Apply Dropout, L2, BatchNorm, Augmentation, and Early Stopping
— individually, then combined — to the same overfit setup, and compare.

Run 02_overfit_baseline.py first; this script reuses its saved history
instead of retraining the baseline.
"""

import sys
from pathlib import Path
from typing import Dict, List, cast

import tensorflow as tf

sys.path.append(str(Path(__file__).parent))

import config
from utils import architecture, data, training, visualization

# %% Same fixed subset as the baseline — fair comparison
x_train, y_train, x_val, y_val = data.get_overfit_subset()
x_train, x_val = data.normalize(x_train), data.normalize(x_val)

# %% Train every config except baseline (reused from 02)
histories = {"baseline": training.load_history("baseline")}

# Separate from `histories` (which holds the full per-epoch curve for plotting).
# For early-stopping configs, restore_best_weights=True reverts the model's
# WEIGHTS to the best epoch — but history.history[...][-1] still reports the
# LAST logged epoch's metrics, not the restored model's actual performance.
# Evaluating explicitly after fit() gives the true restored-model numbers.
final_metrics = {}

for config_name, cfg in config.REGULARIZATION_CONFIGS.items():
    if config_name == "baseline":
        continue

    print(f"\n=== Training config: {config_name} ===")
    model = architecture.build_regularized_cnn(
        dropout=cfg["dropout"],
        l2=cfg["l2"],
        batchnorm=cfg["batchnorm"],
        augment=cfg["augment"],
    )
    model = training.compile_model(
        model,
        learning_rate=config.BASELINE_LEARNING_RATE,
        clipnorm=config.BASELINE_CLIPNORM,
    )

    callbacks: List[tf.keras.callbacks.Callback] = (
        [training.early_stopping_callback()] if cfg["early_stop"] else []
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.OVERFIT_EPOCHS,
        batch_size=config.BASELINE_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    training.save_history(history, config_name)
    histories[config_name] = history.history

    if cfg["early_stop"]:
        # Re-evaluate the restored (best-weights) model — don't trust the
        # last logged epoch, which reflects weights that no longer exist.
        # return_dict=True returns Dict[str, float] at runtime, but the tf
        # stubs don't model that as a distinct overload — they still type
        # the result as float | list[float], so a cast is the accurate fix.
        train_metrics = cast(
            Dict[str, float],
            model.evaluate(x_train, y_train, verbose=0, return_dict=True),
        )
        val_metrics = cast(
            Dict[str, float], model.evaluate(x_val, y_val, verbose=0, return_dict=True)
        )
        final_metrics[config_name] = {
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
        }
    else:
        final_metrics[config_name] = {
            "train_acc": history.history["accuracy"][-1],
            "val_acc": history.history["val_accuracy"][-1],
        }

final_metrics["baseline"] = {
    "train_acc": histories["baseline"]["accuracy"][-1],
    "val_acc": histories["baseline"]["val_accuracy"][-1],
}

# %% Side-by-side comparison plot
visualization.plot_regularization_grid(histories)

# %% Summary table — train/val accuracy gap per config
print(f"{'config':>16} | {'train_acc':>9} | {'val_acc':>9} | {'gap':>6}")
for name, metrics in final_metrics.items():
    train_acc, val_acc = metrics["train_acc"], metrics["val_acc"]
    print(
        f"{name:>16} | {train_acc:>9.3f} | {val_acc:>9.3f} | {train_acc - val_acc:>6.3f}"
    )

print(
    "\nUse these numbers + outputs/regularization_comparison.png to fill in diagnostic_report.md"
)
print("Note: early_stopping/combined rows above reflect the RESTORED model")
print("(model.evaluate after fit), not the last logged epoch in the curve plot.")
