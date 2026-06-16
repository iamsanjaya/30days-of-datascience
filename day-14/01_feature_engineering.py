import warnings
import pandas as pd
from xgboost import XGBRegressor

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    XGB_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
)
from utils.feature_builder import build_all_features
from utils.evaluation import cv_rmsle, print_cv_result, improvement_pct
from utils.encoding import encode_categoricals

warnings.filterwarnings("ignore")

print("=" * 60)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 60)

df_raw = pd.read_csv(RAW_DATA_PATH)

y = df_raw[TARGET_COL]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.fillna("Unknown")

    return encode_categoricals(df)


# baseline
X_base = preprocess(df_raw)

# engineered
df_eng = build_all_features(df_raw)
X_eng = preprocess(df_eng)

# align columns (prevents CV split inconsistencies)
X_eng = X_eng.reindex(columns=X_base.columns.union(X_eng.columns), fill_value=0)
X_base = X_base.reindex(columns=X_eng.columns, fill_value=0)

model = XGBRegressor(**XGB_PARAMS)

base_res = cv_rmsle(model, X_base, y, CV_FOLDS, RANDOM_STATE)
eng_res = cv_rmsle(model, X_eng, y, CV_FOLDS, RANDOM_STATE)

print_cv_result("BASELINE", base_res)
print_cv_result("ENGINEERED", eng_res)

print("\nIMPROVEMENT:", improvement_pct(base_res["mean"], eng_res["mean"]))

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

df_eng.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"[SAVED] engineered dataset -> {PROCESSED_DATA_PATH}")
