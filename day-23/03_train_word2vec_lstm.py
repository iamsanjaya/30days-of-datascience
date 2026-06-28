# %%
"""
03_train_word2vec_lstm.py
Train a Word2Vec embedding on the training corpus, build a Keras Embedding
layer from it, and train an LSTM classifier on top. Uses aggressive_clean text.
"""

import time

import numpy as np
from typing import Any, cast

import config
from utils import architectures, data, embeddings, training

# %%
train_df, val_df, test_df = data.load_splits()
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# %%
print("Training Word2Vec on the training corpus only ...")
start = time.perf_counter()

w2v_model = embeddings.train_word2vec(train_df["aggressive_clean"])
tokenizer = embeddings.build_tokenizer(train_df["aggressive_clean"])
embedding_matrix = embeddings.build_embedding_matrix(w2v_model, tokenizer)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

X_train = embeddings.texts_to_padded_sequences(tokenizer, train_df["aggressive_clean"])
X_val = embeddings.texts_to_padded_sequences(tokenizer, val_df["aggressive_clean"])
X_test = embeddings.texts_to_padded_sequences(tokenizer, test_df["aggressive_clean"])

y_train = train_df["label_id"].to_numpy()
y_val = val_df["label_id"].to_numpy()
y_test = test_df["label_id"].to_numpy()

# %%
model = architectures.build_lstm_model(embedding_matrix)
model.summary()

checkpoint_path = config.MODELS_DIR / "word2vec_lstm.keras"
history = training.train_keras_model(
    model, X_train, y_train, X_val, y_val, checkpoint_path=checkpoint_path
)
train_seconds = time.perf_counter() - start
print(f"Training time (Word2Vec + LSTM fit): {train_seconds:.2f}s")

# %%
training.plot_keras_training_curves(
    history,
    config.OUTPUT_DIR / "training_curves_word2vec_lstm.png",
    title="Word2Vec + LSTM",
)

# IMPORTANT: with restore_best_weights=True, history.history[...][-1] reflects the LAST
# epoch trained, not the restored best-weight epoch. Always call model.evaluate() to get
# the actual performance of the weights now loaded in `model`.
results = cast(Any, model.evaluate(X_val, y_val, verbose=0))

val_loss = float(results[0])
val_acc = float(results[1])
print(
    f"Restored-best-weights validation accuracy: {val_acc:.4f} (val_loss={val_loss:.4f})"
)

# %%
start = time.perf_counter()
test_proba = model.predict(X_test, verbose=0)
inference_seconds = time.perf_counter() - start
test_pred = np.argmax(test_proba, axis=1)

# %%
model_size_mb = training.model_size_on_disk_mb(checkpoint_path)

metrics = training.compute_and_save_metrics(
    model_name="word2vec_lstm",
    y_true=y_test,
    y_pred=test_pred,
    train_seconds=train_seconds,
    inference_seconds=inference_seconds,
    model_size_mb=model_size_mb,
)

print(
    f"\nTest accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f}"
)
print(f"Model size on disk: {model_size_mb:.2f} MB")
print(f"Saved model -> {checkpoint_path}")
print(f"Saved metrics + training curves + confusion matrix -> {config.OUTPUT_DIR}")
