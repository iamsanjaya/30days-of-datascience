# %%
"""
02_train_tfidf_baseline.py
Baseline model: TF-IDF features + Logistic Regression, trained on the
aggressive_clean text column. This is the bar every other model has to beat.
"""

import time

import joblib
from sklearn.linear_model import LogisticRegression

import config
from utils import data, text_features, training

# %%
train_df, val_df, test_df = data.load_splits()
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# %%
vectorizer = text_features.build_tfidf_vectorizer()

start = time.perf_counter()
X_train = text_features.fit_transform_train(vectorizer, train_df["aggressive_clean"])
X_val = text_features.transform(vectorizer, val_df["aggressive_clean"])
X_test = text_features.transform(vectorizer, test_df["aggressive_clean"])
print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

# %%
model = LogisticRegression(
    max_iter=config.LOGREG_MAX_ITER, C=config.LOGREG_C, random_state=config.RANDOM_SEED
)
model.fit(X_train, train_df["label_id"])
train_seconds = time.perf_counter() - start
print(f"Training time (vectorize + fit): {train_seconds:.2f}s")

# %%
val_pred = model.predict(X_val)
val_acc = (val_pred == val_df["label_id"]).mean()
print(f"Validation accuracy: {val_acc:.4f}")

start = time.perf_counter()
test_pred = model.predict(X_test)
inference_seconds = time.perf_counter() - start

# %%
model_path = config.MODELS_DIR / "tfidf_logreg.joblib"
vectorizer_path = config.MODELS_DIR / "tfidf_vectorizer.joblib"
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
model_size_mb = training.model_size_on_disk_mb(
    model_path
) + training.model_size_on_disk_mb(vectorizer_path)

metrics = training.compute_and_save_metrics(
    model_name="tfidf_baseline",
    y_true=test_df["label_id"],
    y_pred=test_pred,
    train_seconds=train_seconds,
    inference_seconds=inference_seconds,
    model_size_mb=model_size_mb,
)

print(
    f"\nTest accuracy: {metrics['accuracy']:.4f} | macro F1: {metrics['macro_f1']:.4f}"
)
print(f"Model size on disk: {model_size_mb:.2f} MB")
print(f"Saved model -> {model_path}")
print(f"Saved metrics + confusion matrix -> {config.OUTPUT_DIR}")
