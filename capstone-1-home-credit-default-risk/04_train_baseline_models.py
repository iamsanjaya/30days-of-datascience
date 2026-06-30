"""
04_train_baseline_models.py — Capstone 1: Home Credit Default Risk

Compares 3 algorithms (Logistic Regression, Random Forest, LightGBM) using
a shared sklearn Pipeline for preprocessing — imputation and encoding are
fit only within CV folds / on the training split, never on the full
dataset, to avoid the leakage risk flagged in 03_feature_engineering.py.

Heavier script — intended to run on Kaggle (GPU/CPU budget), not as a Mac
smoke test, per the local-EDA / Kaggle-training split established from
Day 23 onward.
"""

# %% Imports
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

# %% Load engineered features from 03_feature_engineering.py
df = pd.read_parquet(config.OUTPUT_DIR / "application_train_features.parquet")
print(f"[04_train_baseline_models] Loaded {df.shape}")

# %% Separate features/target, drop the ID column (not predictive)
X = df.drop(columns=[config.ID_COL, config.TARGET_COL])
y = df[config.TARGET_COL]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(
    f"[04_train_baseline_models] {len(numeric_cols)} numeric, "
    f"{len(categorical_cols)} categorical columns"
)

# %% Train/test split — stratified on TARGET, held out until final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)
print(f"[04_train_baseline_models] Train: {X_train.shape}, Test: {X_test.shape}")

# %% Shared preprocessing — fit only inside the Pipeline, never on full data
# Median imputation for numeric (robust to the outliers we saw in EDA, e.g.
# GOODS_PRICE_CREDIT_RATIO's max of 6.67). Most-frequent + one-hot for
# categorical, with unknown categories ignored rather than erroring (in
# case the held-out test split has a category value train didn't see).
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
base_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# %% Model 1 — Logistic Regression (scale-sensitive, needs StandardScaler)
# with_mean=False because OneHotEncoder output is sparse — centering would
# force a dense conversion and likely blow up memory at this column count.
logreg_pipeline = Pipeline(
    steps=[
        ("preprocessor", base_preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
    ]
)

# %% Model 2 — Random Forest (scale-invariant, no scaler needed)
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", base_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1
            ),
        ),
    ]
)

# %% Model 3 — LightGBM (scale-invariant, no scaler needed)
lgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", base_preprocessor),
        ("model", lgb.LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)),
    ]
)

models = {
    "Logistic Regression": logreg_pipeline,
    "Random Forest": rf_pipeline,
    "LightGBM": lgb_pipeline,
}

# %% Cross-validated comparison (Stratified K-Fold, ROC-AUC)
cv = StratifiedKFold(
    n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
)
cv_results = {}

for name, pipeline in models.items():
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring=config.PRIMARY_METRIC, n_jobs=-1
    )
    cv_results[name] = scores
    print(
        f"[04_train_baseline_models] {name}: CV ROC-AUC = "
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )

# %% Fit each model on the full training set, evaluate on the held-out test set
test_results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    test_results[name] = test_auc
    print(f"[04_train_baseline_models] {name} — Held-out test ROC-AUC: {test_auc:.4f}")

# %% Save comparison table for the model card / README
comparison_df = pd.DataFrame(
    {
        "cv_mean_auc": {k: v.mean() for k, v in cv_results.items()},
        "cv_std_auc": {k: v.std() for k, v in cv_results.items()},
        "test_auc": test_results,
    }
)
print(f"\n[04_train_baseline_models] Model comparison:\n{comparison_df}")

comparison_path = config.OUTPUT_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path)
print(f"[04_train_baseline_models] Saved comparison to {comparison_path}")
