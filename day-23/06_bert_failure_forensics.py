# %%
"""
06_bert_failure_forensics.py
Out-of-box challenge for Day 23: find tweets where BERT is confidently wrong,
flag candidate root causes (negation, length, punctuation-as-sarcasm-signal),
then attempt one fix WITHOUT retraining: a negation-aware post-processing rule.

The "likely_root_cause" column is intentionally left blank in the saved CSV —
labeling *why* each failure happened is the actual thinking exercise from the
roadmap and is filled in by hand (in failure_analysis.md / the learning journal),
not auto-generated here.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from utils import bert_utils, data

# %%
train_df, val_df, test_df = data.load_splits()
device = bert_utils.get_device()

checkpoint_dir = config.MODELS_DIR / "distilbert_sentiment"
if not checkpoint_dir.exists():
    raise FileNotFoundError(f"{checkpoint_dir} missing — run 04_train_bert.py first.")


tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

test_ds = bert_utils.TweetDataset(
    test_df["light_clean"], test_df["label_id"], tokenizer
)
test_loader = bert_utils.build_dataloader(
    test_ds, config.BERT_BATCH_SIZE, shuffle=False
)

y_true, y_pred, y_proba = bert_utils.predict_with_probabilities(
    model, test_loader, device
)
confidence = y_proba.max(axis=1)

# %%
results = test_df.copy().reset_index(drop=True)
results["true_label"] = [config.CLASS_NAMES[i] for i in y_true]
results["predicted_label"] = [config.CLASS_NAMES[i] for i in y_pred]
results["confidence"] = confidence

wrong_mask = y_true != y_pred
confidently_wrong_mask = wrong_mask & (
    confidence >= config.FORENSICS_CONFIDENCE_THRESHOLD
)
print(f"Test set size: {len(results)}")
print(f"Wrong predictions: {wrong_mask.sum()}")
print(
    f"Confidently wrong (conf >= {config.FORENSICS_CONFIDENCE_THRESHOLD}): "
    f"{confidently_wrong_mask.sum()}"
)

# %%
_negation_pattern = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in config.NEGATION_WORDS) + r")\b"
)


def has_negation(text: str) -> bool:
    return bool(_negation_pattern.search(str(text).lower()))


failures = results[confidently_wrong_mask].copy()
failures["has_negation_word"] = failures[config.TEXT_COLUMN].apply(has_negation)
failures["text_length_chars"] = failures[config.TEXT_COLUMN].str.len()
failures["likely_root_cause"] = ""  # fill in by hand after reading each tweet

failures = failures.sort_values("confidence", ascending=False).head(
    config.FORENSICS_NUM_FAILURES
)

failures_out = failures[
    [
        config.TEXT_COLUMN,
        "true_label",
        "predicted_label",
        "confidence",
        "has_negation_word",
        "text_length_chars",
        "likely_root_cause",
    ]
]
failures_out.to_csv(config.OUTPUT_DIR / "bert_confident_failures.csv", index=False)
print(
    f"\nSaved {len(failures_out)} confidently-wrong cases to bert_confident_failures.csv"
)
print("Open that file and fill in 'likely_root_cause' for each row by hand —")
print("that judgement call is the actual out-of-box exercise.")

# %%
print("\nTop confidently-wrong examples:")
for _, row in failures_out.head(5).iterrows():
    print(
        f"  [{row['confidence']:.2f}] true={row['true_label']} pred={row['predicted_label']}"
    )
    print(f'    "{row[config.TEXT_COLUMN][:120]}"')

# %%
print("\n--- Fix attempt: negation-aware post-processing rule ---")
print(
    "Rule: if the model predicts 'positive' or 'neutral' but the tweet contains a "
    "negation word, shift one class toward 'negative' (no retraining, just a label rule)."
)


def apply_negation_rule(text: str, predicted_idx: int) -> int:
    if has_negation(text) and predicted_idx != config.LABEL_MAPPING["negative"]:
        # nudge one step toward negative: positive -> neutral, neutral -> negative
        return max(predicted_idx - 1, config.LABEL_MAPPING["negative"])
    return predicted_idx


adjusted_pred = np.array(
    [
        apply_negation_rule(text, pred)
        for text, pred in zip(results[config.TEXT_COLUMN], y_pred)
    ]
)

before_acc = accuracy_score(y_true, y_pred)
after_acc = accuracy_score(y_true, adjusted_pred)
before_f1 = f1_score(y_true, y_pred, average="macro")
after_f1 = f1_score(y_true, adjusted_pred, average="macro")

print(f"Before rule — accuracy: {before_acc:.4f}, macro F1: {before_f1:.4f}")
print(f"After rule  — accuracy: {after_acc:.4f}, macro F1: {after_f1:.4f}")

changed_mask = adjusted_pred != y_pred
fixed = ((y_pred != y_true) & (adjusted_pred == y_true)).sum()
broke = ((y_pred == y_true) & (adjusted_pred != y_true)).sum()
print(
    f"Rule changed {changed_mask.sum()} predictions — fixed {fixed}, broke {broke} "
    f"that were previously correct."
)

pd.DataFrame(
    [
        {"stage": "before_rule", "accuracy": before_acc, "macro_f1": before_f1},
        {"stage": "after_negation_rule", "accuracy": after_acc, "macro_f1": after_f1},
        {
            "stage": "diagnostics",
            "fixed": fixed,
            "broke": broke,
            "changed": int(changed_mask.sum()),
        },
    ]
).to_csv(config.OUTPUT_DIR / "bert_negation_rule_effect.csv", index=False)
print("\nSaved before/after comparison to bert_negation_rule_effect.csv")
