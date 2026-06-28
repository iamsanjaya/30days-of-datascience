"""
utils/text_features.py
TF-IDF vectorization for the classical baseline (02_train_tfidf_baseline.py).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

import config


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
    )


def fit_transform_train(vectorizer: TfidfVectorizer, train_texts):
    """Fit on train only — fitting on val/test would be data leakage."""
    return vectorizer.fit_transform(train_texts)


def transform(vectorizer: TfidfVectorizer, texts):
    return vectorizer.transform(texts)
