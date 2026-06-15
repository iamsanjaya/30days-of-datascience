# Load, encode, and scale Mall Customers dataset

# %% Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from config import DATA_FILE, PROCESSED_DIR, MODELS_DIR, CLUSTERING_FEATURES

# %% Load
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe())

# %% Encode Gender as binary (Male=0, Female=1)
df["Gender_encoded"] = (df["Gender"] == "Female").astype(int)

# %% Select features for clustering
X = df[CLUSTERING_FEATURES].copy()

print(f"\nMissing values:\n{X.isnull().sum()}")
assert X.isnull().sum().sum() == 0, "Unexpected nulls — handle before clustering"

# %% Scale (K-Means is distance-based — scaling is non-negotiable)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=CLUSTERING_FEATURES)

# %% Persist (RUN-SAFE STORAGE)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_scaled_df.to_csv(PROCESSED_DIR / "X_scaled.csv", index=False)
df.to_csv(PROCESSED_DIR / "df_with_encoded.csv", index=False)

joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

print("\nSaved artifacts with run isolation")
print(f"Feature matrix shape: {X_scaled_df.shape}")
