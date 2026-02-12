import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils_io import read_csv_any

OUT = "data/processed/kidney_model.pkl"

def main():
    df, path = read_csv_any(["Chronic_Kidney_Dsease_data.csv"])
    print("Loaded:", path)

    # Candidate columns in YOUR file (based on your dataset)
    candidates = [
        "Age","BloodPressure","SerumCreatinine","BUNLevels","GFR","ACR",
        "ProteinInUrine","BloodGlucoseRandom","HbA1c","Smoking","AlcoholConsumption"
    ]
    cols = [c for c in candidates if c in df.columns]
    if len(cols) < 4:
        raise RuntimeError(f"Kidney dataset missing enough columns. Found only: {cols}")

    if "Diagnosis" not in df.columns:
        raise RuntimeError("Kidney dataset needs a Diagnosis label column.")

    X = df[cols].copy()
    y = df["Diagnosis"].astype(str).str.lower().str.contains("ckd").astype(int)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), cols)
    ])

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred))

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"model": pipe, "features": cols}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
