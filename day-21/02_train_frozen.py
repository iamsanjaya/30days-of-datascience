"""
Day 21 — Step 2: Phase 1 — Frozen Base Training

Builds ResNet50 (ImageNet weights, frozen) + classification head, and
trains only the head. This establishes the "frozen" baseline accuracy
referenced in the standard task deliverable.

Run:
    python 02_train_frozen.py
"""

import config
from utils import architecture, data, training, visualization


def main() -> None:
    train_df, val_df, _ = data.load_splits()

    train_ds = data.build_dataset(train_df, augment=True, shuffle=True)
    val_ds = data.build_dataset(val_df, augment=False, shuffle=False)

    print(f"Building model with frozen {config.BACKBONE} base...")
    model = architecture.build_model(trainable_base=False)
    model = training.compile_model(model, learning_rate=config.FROZEN_LR)
    model.summary()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training for up to {config.FROZEN_EPOCHS} epochs...")
    history = training.train_model(
        model,
        train_ds,
        val_ds,
        epochs=config.FROZEN_EPOCHS,
        checkpoint_path=config.FROZEN_MODEL_PATH,
    )

    training.save_history(history, config.FROZEN_HISTORY_PATH)
    model.save(config.FROZEN_MODEL_PATH)
    print(f"Frozen model saved to {config.FROZEN_MODEL_PATH}")

    visualization.plot_training_curves(
        history,
        title="Phase 1: Frozen Base",
        save_path=config.CURVES_DIR / "phase1_frozen_curves.png",
    )
    print(f"Training curves saved to {config.CURVES_DIR}")

    best_val_acc = max(history["val_accuracy"])
    print(f"Best validation accuracy (frozen): {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
