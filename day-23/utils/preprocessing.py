"""
utils/preprocessing.py
Two distinct cleaning levels, because the three models want different things
from raw tweet text:

- light_clean: BERT has its own subword tokenizer and was pretrained on
  natural text. Stripping punctuation/casing throws away signal it knows
  how to use (e.g. "AMAZING!!" vs "amazing"). We only remove things that
  are pure noise for any model: URLs, @mentions, and excess whitespace.

- aggressive_clean: TF-IDF and the Word2Vec+LSTM path work on a fixed
  vocabulary of discrete tokens, so noise reduction matters more than
  preserving surface form. Lowercase, strip URLs/mentions/hashtag symbols/
  punctuation/digits, remove stopwords, lemmatize.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

_NLTK_RESOURCES = ("stopwords", "wordnet", "omw-1.4")


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora once, quietly, if not already present."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(
                f"corpora/{resource}"
                if resource != "stopwords"
                else "corpora/stopwords"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


_STOPWORDS: set[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_resources()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _get_lemmatizer() -> WordNetLemmatizer:
    global _LEMMATIZER
    if _LEMMATIZER is None:
        ensure_nltk_resources()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


def light_clean(text: str) -> str:
    """Minimal cleaning for BERT: drop URLs/mentions, collapse whitespace, keep everything else."""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def aggressive_clean(text: str) -> str:
    """Full normalization for TF-IDF / Word2Vec+LSTM."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYMBOL_RE.sub("", text)  # keep the word, drop the '#'
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    stop_set = _get_stopwords()
    lemmatizer = _get_lemmatizer()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in text.split()
        if tok not in stop_set and len(tok) >= config.MIN_TOKEN_LENGTH
    ]
    return " ".join(tokens)
