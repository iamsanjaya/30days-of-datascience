# %%
"""Day 20 — 02: Train a deliberately oversized CNN on a tiny subset.

Small data (500 images) + a large model (millions of params) + zero
regularization = textbook overfitting. This run is the baseline every fix
in 03_regularization_comparison.py gets measured against.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from utils import architecture, data, training, visualization

# %% Load the fixed small subset
x_train, y_train, x_val, y_val = data.get_overfit_subset()
x_train, x_val = data.normalize(x_train), data.normalize(x_val)
print(f"Train: {x_train.shape}, Val: {x_val.shape}")

# %% Build and compile the oversized, unregularized model
model = architecture.build_overfit_model()
model = training.compile_model(
    model,
    learning_rate=config.BASELINE_LEARNING_RATE,
    clipnorm=config.BASELINE_CLIPNORM,
)
model.summary()

# %% Train — no early stopping, no augmentation, no dropout, no L2
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=config.OVERFIT_EPOCHS,
    batch_size=config.BASELINE_BATCH_SIZE,
    verbose=2,
)

# %% Save artifacts
training.save_history(history, "baseline")
model.save(config.MODELS_DIR / "overfit_baseline.keras")
visualization.plot_overfit_curves(history.history)

final_gap = history.history["val_loss"][-1] - history.history["loss"][-1]
print(f"\nFinal train/val loss gap: {final_gap:.4f}")
print("The bigger this gap, the worse the overfit — this is what 03 will shrink.")
