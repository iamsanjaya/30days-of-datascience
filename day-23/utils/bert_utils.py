"""
utils/bert_utils.py
PyTorch + HuggingFace plumbing for fine-tuning DistilBERT: device selection
(mps -> cpu fallback on Apple Silicon), a Dataset wrapper, the fine-tuning
loop, and a prediction helper reused by 06_bert_failure_forensics.py.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import config


def get_device() -> torch.device:
    """Try each device in config.BERT_DEVICE_PREFERENCE, in order, and use the first available.

    Resolves to "cuda" on Kaggle (NVIDIA GPU), "mps" on Apple Silicon locally,
    or "cpu" as the universal fallback — same code runs in both environments.
    """
    for name in config.BERT_DEVICE_PREFERENCE:
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = config.BERT_MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_tokenizer_and_model(num_labels: int = config.NUM_CLASSES):
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BERT_MODEL_NAME, num_labels=num_labels
    )
    return tokenizer, model


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fine_tune(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.BERT_EPOCHS,
    lr: float = config.BERT_LEARNING_RATE,
    warmup_ratio: float = config.BERT_WARMUP_RATIO,
) -> dict:
    """Fine-tune DistilBERT, returning a history dict (train/val loss + accuracy per epoch)."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(
            f"[BERT] epoch {epoch + 1}/{epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return history


@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader: DataLoader, device: torch.device):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def predict_with_probabilities(model, loader: DataLoader, device: torch.device):
    """Returns (y_true, y_pred, y_proba) as numpy arrays — used for metrics + forensics."""
    model.to(device)
    model.eval()
    all_true, all_pred, all_proba = [], [], []
    for batch in loader:
        labels = batch["labels"]
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_true.append(labels.numpy())
        all_pred.append(preds)
        all_proba.append(probs)
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


def save_checkpoint(model, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
