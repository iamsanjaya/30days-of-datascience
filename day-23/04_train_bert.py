# %%
"""
04_train_bert.py
Fine-tune DistilBERT (HuggingFace + PyTorch) for 3-class sentiment.
Uses light_clean text — BERT's subword tokenizer benefits from natural
casing/punctuation rather than the aggressive normalization used for TF-IDF/LSTM.
"""

import time

import matplotlib.pyplot as plt

import config
from utils import bert_utils, data, training

# %%
train_df, val_df, test_df = data.load_splits()
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

device = bert_utils.get_device()
print(f"Using device: {device}")

# %%
tokenizer, model = bert_utils.load_tokenizer_and_model()

train_ds = bert_utils.TweetDataset(
    train_df["light_clean"], train_df["label_id"], tokenizer
)
val_ds = bert_utils.TweetDataset(val_df["light_clean"], val_df["label_id"], tokenizer)
test_ds = bert_utils.TweetDataset(
    test_df["light_clean"], test_df["label_id"], tokenizer
)

train_loader = bert_utils.build_dataloader(
    train_ds, config.BERT_BATCH_SIZE, shuffle=True
)
val_loader = bert_utils.build_dataloader(val_ds, config.BERT_BATCH_SIZE, shuffle=False)
test_loader = bert_utils.build_dataloader(
    test_ds, config.BERT_BATCH_SIZE, shuffle=False
)

# %%
start = time.perf_counter()
history = bert_utils.fine_tune(model, train_loader, val_loader, device)
train_seconds = time.perf_counter() - start
print(f"Training time (fine-tune): {train_seconds:.2f}s")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("DistilBERT — Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(history["val_accuracy"], label="val")
axes[1].set_title("DistilBERT — Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(config.OUTPUT_DIR / "training_curves_bert.png", dpi=config.FIGURE_DPI)
plt.close(fig)

# %%
start = time.perf_counter()
y_true, y_pred, y_proba = bert_utils.predict_with_probabilities(
    model, test_loader, device
)
inference_seconds = time.perf_counter() - start

# %%
checkpoint_dir = config.MODELS_DIR / "distilbert_sentiment"
bert_utils.save_checkpoint(model, tokenizer, checkpoint_dir)
model_size_mb = training.model_size_on_disk_mb(checkpoint_dir)

metrics = training.compute_and_save_metrics(
    model_name="bert",
    y_true=y_true,
    y_pred=y_pred,
    train_seconds=train_seconds,
    inference_seconds=inference_seconds,
    model_size_mb=model_size_mb,
)

print(
    f"\nTest accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f}"
)
print(f"Model size on disk: {model_size_mb:.2f} MB")
print(f"Saved checkpoint -> {checkpoint_dir}")
print(f"Saved metrics + training curves + confusion matrix -> {config.OUTPUT_DIR}")
