
import numpy as np
import pandas as pd
from .features import enrich_proprietary
from .config import REQUIRED_COLUMNS, NUM_FEATURES, CAT_FEATURES

def load_sample_data() -> pd.DataFrame:
    return pd.read_csv("sample_data/synthetic_nola_roofs.csv")

def validate_and_prepare(df: pd.DataFrame):
    issues = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str)
    if "fortified" in df.columns:
        df["fortified"] = df["fortified"].fillna(0).astype(int).clip(0,1)
    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat","lon"])
    df = df.copy()
    df = enrich_proprietary(df)
    # Ensure downstream feature pipelines always see expected columns
    for col in NUM_FEATURES + CAT_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df, issues
