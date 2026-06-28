"""
utils/embeddings.py
Tokenization/padding for the LSTM input, plus training a Word2Vec model on
the training corpus and converting it into a Keras-compatible embedding matrix.
"""

import numpy as np
from gensim.models import Word2Vec
from keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

import config


def build_tokenizer(train_texts) -> Tokenizer:
    tokenizer = Tokenizer(num_words=config.LSTM_MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def texts_to_padded_sequences(tokenizer: Tokenizer, texts) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences, maxlen=config.LSTM_MAX_LEN, padding="post", truncating="post"
    )


def train_word2vec(train_texts) -> Word2Vec:
    """Train Word2Vec on the (already aggressively-cleaned) training corpus only."""
    tokenized_sentences = [text.split() for text in train_texts]
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=config.W2V_EMBEDDING_DIM,
        window=config.W2V_WINDOW,
        min_count=config.W2V_MIN_COUNT,
        sg=config.W2V_SG,
        epochs=config.W2V_EPOCHS,
        seed=config.RANDOM_SEED,
        workers=1,  # workers>1 is non-deterministic even with a fixed seed
    )
    return model


def build_embedding_matrix(w2v_model: Word2Vec, tokenizer: Tokenizer) -> np.ndarray:
    """Rows aligned to tokenizer word indices; OOV/unseen words get zero vectors."""
    vocab_size = min(config.LSTM_MAX_VOCAB, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, config.W2V_EMBEDDING_DIM))

    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
        # else: leave as zero vector (word never seen by Word2Vec, e.g. rare/OOV token)

    return embedding_matrix
