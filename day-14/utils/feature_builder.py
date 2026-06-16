import numpy as np
import pandas as pd
from config import LOG_TRANSFORM_COLS

ENGINEERED_FEATURE_NAMES = [
    "TotalSF",
    "Age",
    "RemodAge",
    "HasPool",
    "HasGarage",
]


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bsmt = df.get("Total Bsmt SF", pd.Series(0, index=df.index)).fillna(0)
    first_flr = df.get("1st Flr SF", pd.Series(0, index=df.index)).fillna(0)
    second_flr = df.get("2nd Flr SF", pd.Series(0, index=df.index)).fillna(0)

    df["TotalSF"] = bsmt + first_flr + second_flr

    df["Age"] = 2026 - df["Year Built"]
    df["RemodAge"] = 2026 - df["Year Remod/Add"]

    df["HasPool"] = (
        df.get("Pool Area", pd.Series(0, index=df.index)).fillna(0) > 0
    ).astype(np.int8)
    df["HasGarage"] = (
        df.get("Garage Area", pd.Series(0, index=df.index)).fillna(0) > 0
    ).astype(np.int8)

    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f"log_{col.lower().replace(' ', '_')}"] = np.log1p(df[col].fillna(0))

    return df
