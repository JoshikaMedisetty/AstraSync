import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils_io import read_csv_any
from utils_columns import map_human_vitals

OUT = "data/processed/heart_model.pkl"

def main():
    df_raw, path = read_csv_any(["human_vital_signs_dataset_2024.csv"])
    print("Loaded:", path)

    df = map_human_vitals(df_raw)

    feats = ["age","gender","bp_sys","bp_dia","resting_hr","spo2_avg","height_m","weight_kg"]
    X = df[feats].copy()

    # rule-based label for heart risk (binary)
    y = ((df["bp_sys"] >= 140) | (df["bp_dia"] >= 90) | (df["resting_hr"] >= 95) | (df["spo2_avg"] < 94)).astype(int)

    cat = ["gender"]
    num = [c for c in feats if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # drop rows with no bp_sys and no hr
    keep = df["bp_sys"].notna() | df["resting_hr"].notna()
    X = X[keep]
    y = y[keep]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred))

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"model": pipe, "features": feats}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
