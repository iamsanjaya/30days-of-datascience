# Day 23 — June 25, 2026

## What was built

A three-way sentiment classifier comparison on the Twitter US Airline
Sentiment dataset: TF-IDF + Logistic Regression baseline, Word2Vec embeddings
feeding an LSTM, and a fine-tuned DistilBERT model. Two distinct text-cleaning
pipelines feed these (light for BERT, aggressive for TF-IDF/LSTM), and a
shared `utils/training.py` contract (`compute_and_save_metrics`) keeps the
three model scripts comparable in `05_compare_models.py`.

Full pipeline ran end-to-end on Kaggle (GPU, full dataset, real epoch counts)
after building it and smoke-testing it locally first.

- TF-IDF baseline test accuracy / macro F1: **0.7719 / 0.6760**
- Word2Vec + LSTM test accuracy / macro F1: **0.7811 / 0.7001**
- DistilBERT test accuracy / macro F1: **0.8408 / 0.7862**
- Training time / inference time / model size:

  | Model           | Train Time | Inference (ms/1000) | Size on Disk |
  | --------------- | ---------- | ------------------- | ------------ |
  | TF-IDF + LogReg | 3.34s      | 0.0003              | 0.61 MB      |
  | Word2Vec + LSTM | 23.59s     | 0.2143              | 10.36 MB     |
  | DistilBERT      | 214.09s    | 1.9616              | 256.11 MB    |

  Full table in `outputs/model_comparison.csv`.

## The out-of-box challenge result

- Number of confidently-wrong BERT predictions found: **265** (out of 344
  total misclassifications on a 2,161-row test set — 8 saved to
  `outputs/bert_confident_failures.csv` for manual review)
- Root causes identified after manually reading the failures: the clearest
  pattern across the top confidently-wrong cases is **mixed-sentiment /
  backhanded-compliment tweets** — e.g. a tweet ending "very frustrated" that
  was originally labeled positive, and several complaint-toned tweets
  ("#epicfail", "horrible") that were actually labeled neutral by the human
  annotators. BERT is reading surface negative tone and missing the
  annotator's more nuanced overall read. Negation by itself doesn't explain
  most of these — the words present are negative-coded, but the _true_ label
  isn't, which is the opposite of a simple negation problem.
- Negation post-processing rule effect:
  - Before rule — accuracy: 0.8408, macro F1: 0.7862
  - After rule — accuracy: 0.8376, macro F1: 0.7788
  - Rule changed 43 predictions: fixed 14, broke 21 previously-correct ones
  - Net effect: **worse**, not better. Full case-by-case data in
    `outputs/bert_negation_rule_effect.csv`.

## What surprised me

The negation rule made things worse, and the reason why is the actual
finding: I expected negation handling to be the lever that fixes BERT's
mistakes, but the failure cases I read were mostly mixed-sentiment or
backhanded tone, not literal negation. A blunt rule that shifts toward
"negative" whenever a negation word appears ends up punishing tweets that
already used negation correctly and were already classified right — it has
no way to distinguish "this negation flips the sentiment" from "this
negation was already accounted for by the model." Cheap post-processing
rules built on a surface heuristic can't repair a model's deeper
misunderstanding of tone.

## What I don't fully understand yet

Why the negation rule specifically broke 21 previously-correct predictions
rather than just failing to fix new ones — i.e. which of those 21 cases had
negation words but were already correctly classified, and whether a more
targeted version of the rule (e.g. only applying it when the negation word
is near the end of the tweet, or combined with a confidence threshold on the
original prediction) would have avoided breaking them. Haven't dug into the
per-case breakdown of those 21 yet.

## GitHub commit made: ✅

`day-23: bow-to-bert comparison + bert failure forensics`

## Tomorrow's priority

Start Capstone #1 (Days 24–25) — Structured Data ML Project. Pick a Kaggle
dataset, do EDA + feature engineering, compare 3+ models, tune with Optuna,
add SHAP explanations, write a Risk Report arguing against deployment, and
submit to the Kaggle leaderboard. Commit: `capstone-1: [project name] —
complete with risk report`.
