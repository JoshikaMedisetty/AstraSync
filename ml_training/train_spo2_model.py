import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils_io import read_csv_any
from utils_columns import map_wearable_spo2

OUT = "data/processed/spo2_model.pkl"

def main():
    df_raw, path = read_csv_any(["wearable_sports_health_dataset*.csv"])
    print("Loaded:", path)

    df = map_wearable_spo2(df_raw)

    feats = ["spo2_avg","resting_hr","steps"]
    X = df[feats].copy()

    spo2 = pd.to_numeric(df["spo2_avg"], errors="coerce")
    hr = pd.to_numeric(df["resting_hr"], errors="coerce")

    # risk label (binary): yellow/red=1
    y = ((spo2 < 94) | (hr > 95)).astype(int)
    # drop rows where spo2 is missing
    keep = spo2.notna()
    X = X[keep]
    y = y[keep]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), feats)
    ])

    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred))

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"model": pipe, "features": feats}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
