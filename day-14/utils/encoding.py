import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = pd.Series(
            le.fit_transform(df[col].astype(str)), index=df.index, name=col
        )

    return df
