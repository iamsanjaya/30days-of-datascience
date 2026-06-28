# %%
"""
05_compare_models.py
Load the metrics JSON written by 02/03/04 and build the comparison table +
chart that documents the performance gap between TF-IDF, Word2Vec+LSTM, and BERT.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

import config

MODEL_KEYS = ["tfidf_baseline", "word2vec_lstm", "bert"]
DISPLAY_NAMES = {
    "tfidf_baseline": "TF-IDF + LogReg",
    "word2vec_lstm": "Word2Vec + LSTM",
    "bert": "DistilBERT",
}

# %%
rows = []
for key in MODEL_KEYS:
    path = config.OUTPUT_DIR / f"metrics_{key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run the corresponding training script first."
        )
    with open(path) as f:
        m = json.load(f)
    rows.append(
        {
            "model": DISPLAY_NAMES[key],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "train_seconds": m["train_seconds"],
            "inference_ms_per_1000": m["inference_seconds_per_1000"],
            "model_size_mb": m["model_size_mb"],
        }
    )

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(config.OUTPUT_DIR / "model_comparison.csv", index=False)
print(comparison_df.to_string(index=False))

# %%
print("\nMarkdown table (paste into README.md):\n")
print(comparison_df.to_markdown(index=False, floatfmt=".4f"))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].bar(comparison_df["model"], comparison_df["accuracy"], color="#2980b9")
axes[0].set_title("Test Accuracy")
axes[0].set_ylim(0, 1)

axes[1].bar(comparison_df["model"], comparison_df["train_seconds"], color="#e67e22")
axes[1].set_title("Training Time (s)")

axes[2].bar(comparison_df["model"], comparison_df["model_size_mb"], color="#8e44ad")
axes[2].set_title("Model Size (MB) — log scale")
axes[2].set_yscale("log")

for ax in axes:
    ax.tick_params(axis="x", rotation=20)

fig.suptitle("TF-IDF vs Word2Vec+LSTM vs BERT")
fig.tight_layout()
fig.savefig(config.OUTPUT_DIR / "model_comparison.png", dpi=config.FIGURE_DPI)
plt.close(fig)
print(f"\nSaved model_comparison.csv + model_comparison.png to {config.OUTPUT_DIR}")
