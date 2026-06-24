"""
Day 21 — Step 3: Phase 2 — Fine-Tuning

Loads the trained frozen model, unfreezes the last config.UNFREEZE_LAYERS
layers of the backbone (config.BACKBONE), recompiles with a much smaller learning
rate, and continues training. This is where the "accuracy jump from
frozen vs fine-tuned" deliverable comes from.

Run:
    python 03_finetune.py
"""

import tensorflow as tf

import config
from utils import architecture, data, training, visualization


def main() -> None:
    train_df, val_df, _ = data.load_splits()

    train_ds = data.build_dataset(train_df, augment=True, shuffle=True)
    val_ds = data.build_dataset(val_df, augment=False, shuffle=False)

    print(f"Loading frozen model from {config.FROZEN_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.FROZEN_MODEL_PATH)

    print(
        f"Unfreezing last {config.UNFREEZE_LAYERS} layers of {config.BACKBONE} base..."
    )
    model = architecture.unfreeze_top_layers(model, n_layers=config.UNFREEZE_LAYERS)

    trainable_count = sum(
        1 for layer in architecture.get_base_model(model).layers if layer.trainable
    )
    print(
        f"Trainable base layers: {trainable_count} / {len(architecture.get_base_model(model).layers)}"
    )

    model = training.compile_model(model, learning_rate=config.FINETUNE_LR)
    model.summary()

    print(
        f"Fine-tuning for up to {config.FINETUNE_EPOCHS} epochs at lr={config.FINETUNE_LR}..."
    )
    history = training.train_model(
        model,
        train_ds,
        val_ds,
        epochs=config.FINETUNE_EPOCHS,
        checkpoint_path=config.FINETUNED_MODEL_PATH,
    )

    training.save_history(history, config.FINETUNE_HISTORY_PATH)
    model.save(config.FINETUNED_MODEL_PATH)
    print(f"Fine-tuned model saved to {config.FINETUNED_MODEL_PATH}")

    visualization.plot_training_curves(
        history,
        title="Phase 2: Fine-Tuned",
        save_path=config.CURVES_DIR / "phase2_finetune_curves.png",
    )

    frozen_history = training.load_history(config.FROZEN_HISTORY_PATH)
    visualization.plot_combined_curves(
        frozen_history,
        history,
        save_path=config.CURVES_DIR / "combined_frozen_vs_finetune_curves.png",
    )
    print(f"Training curves saved to {config.CURVES_DIR}")

    best_val_acc = max(history["val_accuracy"])
    print(f"Best validation accuracy (fine-tuned): {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
