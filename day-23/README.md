# Day 23 — NLP: From Bag-of-Words to BERT

Twitter Airline Sentiment classification, three ways: TF-IDF + Logistic Regression baseline, Word2Vec embeddings + LSTM, and fine-tuned DistilBERT. Includes a model comparison and a manual forensics pass on BERT's confidently-wrong predictions.

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (`crowdflower/twitter-airline-sentiment` on Kaggle) — 14,640 tweets directed at major US airlines, labeled negative / neutral / positive.

- Raw rows: 14,640
- After dropping nulls/duplicates: 14,427
- After dropping rows empty post-cleaning: 14,402
- Split — train: 10,404 · val: 1,837 · test: 2,161

Class distribution is imbalanced toward negative across all splits (train: 6,553 negative / 2,199 neutral / 1,652 positive), which is expected for airline complaint data and is reflected in the macro-F1 metric used throughout.

## Project structure

```
day-23/
├── 01_data_preparation.py           # Data cleaning, preprocessing, train/val/test split
├── 02_train_tfidf_baseline.py       # TF-IDF + Logistic Regression baseline model
├── 03_train_word2vec_lstm.py        # Word2Vec embedding generation and LSTM training
├── 04_train_bert.py                 # DistilBERT fine-tuning for sentiment classification
├── 05_compare_models.py             # Compare TF-IDF, LSTM, and BERT performance
├── 06_bert_failure_forensics.py     # Analyze BERT misclassifications and failure patterns
├── README.md                        # Project documentation and results summary
├── config.py                        # Central configuration, paths, and hyperparameters
├── data
│   ├── processed
│   │   ├── test.csv                 # Held-out test dataset
│   │   ├── train.csv                # Training dataset
│   │   └── val.csv                  # Validation dataset
│   └── raw
│       └── tweets.csv               # Original Twitter US Airline Sentiment dataset
├── models
│   ├── distilbert_sentiment         # Fine-tuned DistilBERT model and tokenizer files
│   │   ├── config.json              # DistilBERT configuration
│   │   ├── model.safetensors        # Fine-tuned model weights
│   │   ├── tokenizer.json           # Tokenizer vocabulary and settings
│   │   └── tokenizer_config.json    # Tokenizer configuration
│   ├── tfidf_logreg.joblib          # Trained Logistic Regression model
│   ├── tfidf_vectorizer.joblib      # Fitted TF-IDF vectorizer
│   └── word2vec_lstm.keras          # Trained Word2Vec-LSTM model
├── outputs
│   ├── bert_confident_failures.csv      # High-confidence BERT prediction errors
│   ├── bert_negation_rule_effect.csv    # Negation analysis results
│   ├── class_distribution.png           # Dataset class distribution visualization
│   ├── confusion_matrix_bert.png        # BERT confusion matrix
│   ├── confusion_matrix_tfidf_baseline.png
│   ├── confusion_matrix_word2vec_lstm.png
│   ├── metrics_bert.json                # BERT evaluation metrics
│   ├── metrics_tfidf_baseline.json      # TF-IDF baseline metrics
│   ├── metrics_word2vec_lstm.json       # LSTM evaluation metrics
│   ├── model_comparison.csv             # Consolidated model comparison table
│   ├── model_comparison.png             # Model comparison visualization
│   ├── training_curves_bert.png         # BERT training and validation curves
│   └── training_curves_word2vec_lstm.png
└── utils
    ├── __init__.py                  # Package initialization
    ├── architectures.py             # Neural network architectures
    ├── bert_utils.py                # BERT training and inference helpers
    ├── data.py                      # Data loading utilities
    ├── embeddings.py                # Word2Vec embedding utilities
    ├── preprocessing.py             # Text cleaning and preprocessing pipeline
    ├── text_features.py             # TF-IDF and feature engineering utilities
    └── training.py                  # Shared training and evaluation functions


learning-journal/
   └── day-23.md
```

## How to run

**Locally** (correctness check, small/fast settings via `config.py` environment auto-detection):

```bash
source ml-env/bin/activate
python 01_data_preparation.py
python 02_train_tfidf_baseline.py
python 03_train_word2vec_lstm.py
python 04_train_bert.py
python 05_compare_models.py
python 06_bert_failure_forensics.py
```

**On Kaggle** (full training run, GPU):

1. Attach the `crowdflower/twitter-airline-sentiment` dataset to a new notebook.
2. Enable GPU accelerator.
3. Install pinned dependencies: `pip install --upgrade gensim` (Kaggle's stock numpy/scipy already satisfy torch/tensorflow/transformers — no downgrades needed; see Known Issues).
4. `%%writefile` each `utils/*.py` file into the notebook (create the `utils/` directory first — see Known Issues).
5. `%%writefile` and run each numbered script in order.
6. **Save Version → Save & Run All (Commit)** to get a reproducible run and a downloadable Output tab.

`config.py` auto-detects the Kaggle environment (`KAGGLE_KERNEL_RUN_TYPE` / `/kaggle/input` presence) and switches paths, dataset filename casing, and training settings (full data + full epochs vs. local small-sample + few-epoch) automatically — no manual edits or CLI flags needed.

## Results

| Model           | Test Accuracy | Macro F1 | Train Time | Inference (ms/1000) | Size on Disk |
| --------------- | ------------- | -------- | ---------- | ------------------- | ------------ |
| TF-IDF + LogReg | 0.7719        | 0.6760   | 3.34s      | 0.0003              | 0.61 MB      |
| Word2Vec + LSTM | 0.7811        | 0.7001   | 23.59s     | 0.2143              | 10.36 MB     |
| DistilBERT      | 0.8408        | 0.7862   | 214.09s    | 1.9616              | 256.11 MB    |

DistilBERT wins clearly on accuracy and macro F1, at a real cost: ~64x the training time of the LSTM and ~420x the disk size of the TF-IDF baseline. This tradeoff — and not just the win — is the point of the comparison.

### BERT failure forensics

Out of 2,161 test examples, BERT misclassified 344 (15.9%), of which 265 were "confidently wrong" (predicted-class probability ≥ 0.70). The clearest failure pattern in the top confidently-wrong cases is **backhanded compliments and mixed-sentiment tweets labeled neutral or positive by annotators but read as negative by the model** — e.g. a tweet ending in "very frustrated" but originally rated positive, and several "epicfail"/complaint-toned tweets that were actually labeled neutral.

**Fix attempt:** a negation-aware post-processing rule (shift prediction toward negative if a negation word is present and the predicted class isn't already negative). Result: **made things slightly worse**, not better.

- Before rule — accuracy: 0.8408, macro F1: 0.7862
- After rule — accuracy: 0.8376, macro F1: 0.7788
- Rule changed 43 predictions: fixed 14, broke 21 previously-correct predictions.

This is a genuine and useful negative result: the model's actual failure mode is more about sarcasm, backhanded phrasing, and mixed sentiment than literal negation, so a blunt negation rule isn't the right lever. See `outputs/bert_confident_failures.csv` and `outputs/bert_negation_rule_effect.csv` for the full case-by-case data, and `bert_confident_failures.csv`'s `likely_root_cause` column for manual annotation.

## Known issues encountered (Kaggle setup)

- **gensim/scipy/numpy version conflict.** Kaggle's base image ships numpy 2.x; the locally-pinned `gensim==4.3.2` requires `scipy<1.13`/`numpy<2.0`. Attempting to force those versions on Kaggle broke tensorflow's numpy ABI compatibility and triggered cascading `KeyError`/`AttributeError` import failures, made worse by Kaggle's pre-built image having ~300+ interdependent packages already pinned (unlike a clean local venv). **Fix: do not downgrade.** `pip install --upgrade gensim` (resolves to 4.4.0+) supports numpy 2.x natively and imports clean against Kaggle's stock environment. Local env keeps `gensim==4.3.2`; both are gensim 4.x and API-compatible for this project's Word2Vec usage.
- **`%%writefile utils/<file>.py` fails with `FileNotFoundError` if `utils/` doesn't exist yet.** `%%writefile` does not create parent directories. Fix: run `os.makedirs("utils", exist_ok=True)` in its own cell before any `%%writefile utils/...` cell.
- **Dataset mount path is deeper than the typical `/kaggle/input/<slug>/` pattern.** This dataset mounts at `/kaggle/input/datasets/organizations/crowdflower/twitter-airline-sentiment/`, and the file is named `Tweets.csv` (capital T) rather than the locally-used lowercase `tweets.csv`. `config.py`'s `IS_KAGGLE` branch resolves both differences automatically.

## Local environment

- `ml-env` (Python 3.11), `gensim==4.3.2`, `scipy<1.13`
- Two local bugs fixed during smoke testing: a `gensim`/`scipy` incompatibility, and a device-mismatch bug in `utils/bert_utils.py` (`predict_with_probabilities()` and `evaluate_loss_and_accuracy()` now call `model.to(device)` before use)

## Notes on large files

`models/distilbert_sentiment/` is ~256 MB — over GitHub's 100MB file limit. Either use Git LFS or `.gitignore` the model weights and rely on `04_bert_finetune.py` to regenerate them; this repo currently `.gitignore`s the weights and commits only metrics, plots, and code.
