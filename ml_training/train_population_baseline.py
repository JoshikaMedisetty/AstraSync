import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from utils_io import read_csv_any
from utils_columns import map_human_vitals

OUT = "data/processed/population_baseline.pkl"

def main():
    df_raw, path = read_csv_any(["human_vital_signs_dataset_2024.csv"])
    print("Loaded:", path)

    df = map_human_vitals(df_raw)

    # Targets we can baseline
    targets = ["bp_sys", "bp_dia", "resting_hr", "spo2_avg"]
    feats = ["age", "gender", "height_m", "weight_kg"]

    # Keep only rows with at least 2 targets present
    good = df[targets].notna().sum(axis=1) >= 2
    df = df.loc[good].copy()

    if df.empty:
        raise RuntimeError("No usable rows after filtering. Check dataset columns.")

    X = df[feats]
    Y = df[targets]

    cat = ["gender"]
    num = ["age", "height_m", "weight_kg"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    reg = RandomForestRegressor(n_estimators=250, random_state=42)
    model = Pipeline([
        ("pre", pre),
        ("reg", MultiOutputRegressor(reg))
    ])

    model.fit(X, Y)

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"model": model, "features": feats, "targets": targets}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
