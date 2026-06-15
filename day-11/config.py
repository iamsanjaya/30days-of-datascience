# %%
# Both 01_xgboost_titanic.py and 02_metric_comparison.py import from here.
# Any change to preprocessing or hyperparameters propagates to both scripts
# automatically - no risk of the two files drifting out of sync.

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Paths
DATA_RAW = (
    "/Users/sanjayachaudhary/Desktop/projects/day-11/data/raw/titanic_uncleaned.csv"
)
DATA_PROCESSED = Path("/Users/sanjayachaudhary/Desktop/projects/day-11/data/processed/")
OUTPUTS = Path(__file__).parent / "outputs"
DATA_PROCESSED.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# Experiment constants
RANDOM_STATE = 42
N_FOLDS = 5

# Columns dropped before modelling
# "alive" — string duplicate of the "survived" target
# "who" — derived from sex + age (leakage)
# "adult_male" — derived from sex + age (leakage)
# "alone" — derived from sibsp + parch (leakage)
# "embark_town" — string duplicate of "embarked"
# "deck" — >75% missing: too sparse to impute reliably

COLS_TO_DROP = ["Name", "Ticket", "Cabin", "PassengerId"]


def load_and_preprocess():
    """
    Load Titanic from local CSV, clean, encode, and save to processed/.
    Both 01_xgboost_titanic and 02_metric_comparison scripts read from the saved CSV directly.
    """
    df = pd.read_csv(DATA_RAW)
    df = df.drop(columns=COLS_TO_DROP)
    print(df.columns)

    # %%
    # Missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode categoricals
    le = LabelEncoder()
    df["Sex"] = np.array(le.fit_transform(df["Sex"]), dtype=int)  # female=0, male=1
    df["Embarked"] = np.array(
        le.fit_transform(df["Embarked"]), dtype=int
    )  # C=0, Q=1, S=2

    # Convert any remaining pandas categoricals
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].cat.codes

    # Save cleaned data
    df.to_csv(DATA_PROCESSED / "titanic_cleaned.csv", index=False)
    print(f"[RESULT] Cleaned data saved → {DATA_PROCESSED / 'titanic_cleaned.csv'}")

    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def make_skf():
    """Return the shared StratifiedKFold splitter."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_xgb_model():
    """
    Return a fresh XGBClassifier with the Day 11 hyperparameters.
    Called by both scripts so the model definition stays in one place.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
