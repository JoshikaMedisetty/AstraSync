import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils_io import read_csv_any
from utils_columns import map_sleep_lifestyle

OUT = "data/processed/sleep_model.pkl"

def main():
    df_raw, path = read_csv_any(["sleep_health_lifestyle_dataset.csv", "sleep_health_and_lifestyle_dataset.csv"])
    print("Loaded:", path)

    df = map_sleep_lifestyle(df_raw)

    # features aligned with your app
    feats = ["age","gender","sleep_hours","resting_hr","steps","stress_level","quality_sleep"]
    X = df[feats].copy()

    if df["_sleep_disorder"].isna().all():
        raise RuntimeError("Sleep Disorder label not found in this sleep dataset.")
    y = df["_sleep_disorder"].astype(int)

    cat = ["gender"]
    num = [c for c in feats if c not in cat]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
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
