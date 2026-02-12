import pandas as pd
import numpy as np

def standardize_gender(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "1"]: return "Male"
    if s in ["f", "female", "woman", "0"]: return "Female"
    return "Unknown"

def parse_bp_text(s):
    # "120/80" -> (120, 80)
    if pd.isna(s): return (np.nan, np.nan)
    t = str(s).strip()
    if "/" in t:
        a, b = t.split("/", 1)
        return (pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"))
    return (np.nan, np.nan)

def map_human_vitals(df):
    # human_vital_signs_dataset_2024.csv
    out = pd.DataFrame()
    out["age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    out["gender"] = df.get("Gender").map(standardize_gender) if "Gender" in df.columns else "Unknown"
    # height might be "Height (m)" OR "Height (cm)" depending on dataset
    if "Height (m)" in df.columns:
        out["height_m"] = pd.to_numeric(df["Height (m)"], errors="coerce")
    elif "Height (cm)" in df.columns:
        out["height_m"] = pd.to_numeric(df["Height (cm)"], errors="coerce") / 100.0
    else:
        out["height_m"] = np.nan

    if "Weight (kg)" in df.columns:
        out["weight_kg"] = pd.to_numeric(df["Weight (kg)"], errors="coerce")
    else:
        out["weight_kg"] = np.nan

    out["resting_hr"] = pd.to_numeric(df.get("Heart Rate"), errors="coerce")
    out["spo2_avg"] = pd.to_numeric(df.get("Oxygen Saturation"), errors="coerce")
    out["bp_sys"] = pd.to_numeric(df.get("Systolic Blood Pressure"), errors="coerce")
    out["bp_dia"] = pd.to_numeric(df.get("Diastolic Blood Pressure"), errors="coerce")
    return out

def map_sleep_lifestyle(df):
    # sleep_health_lifestyle_dataset.csv (400 rows) OR similar
    out = pd.DataFrame()
    out["age"] = pd.to_numeric(df.get("Age"), errors="coerce")
    out["gender"] = df.get("Gender").map(standardize_gender) if "Gender" in df.columns else "Unknown"

    # sleep hours
    if "Sleep Duration (hours)" in df.columns:
        out["sleep_hours"] = pd.to_numeric(df["Sleep Duration (hours)"], errors="coerce")
    elif "Sleep Duration" in df.columns:
        out["sleep_hours"] = pd.to_numeric(df["Sleep Duration"], errors="coerce")
    else:
        out["sleep_hours"] = np.nan

    # HR
    if "Heart Rate (bpm)" in df.columns:
        out["resting_hr"] = pd.to_numeric(df["Heart Rate (bpm)"], errors="coerce")
    elif "Heart Rate" in df.columns:
        out["resting_hr"] = pd.to_numeric(df["Heart Rate"], errors="coerce")
    else:
        out["resting_hr"] = np.nan

    # Steps
    if "Daily Steps" in df.columns:
        out["steps"] = pd.to_numeric(df["Daily Steps"], errors="coerce")
    else:
        out["steps"] = np.nan

    # Stress & Quality
    out["stress_level"] = pd.to_numeric(df.get("Stress Level"), errors="coerce")
    out["quality_sleep"] = pd.to_numeric(df.get("Quality of Sleep"), errors="coerce")

    # BP string
    if "Blood Pressure (systolic/diastolic)" in df.columns:
        sys_dia = df["Blood Pressure (systolic/diastolic)"].apply(parse_bp_text)
        out["bp_sys"] = [t[0] for t in sys_dia]
        out["bp_dia"] = [t[1] for t in sys_dia]
    else:
        out["bp_sys"] = np.nan
        out["bp_dia"] = np.nan

    # Sleep disorder label (if exists)
    if "Sleep Disorder" in df.columns:
        y = df["Sleep Disorder"].fillna("None").astype(str).str.lower()
        out["_sleep_disorder"] = (y != "none").astype(int)
    else:
        out["_sleep_disorder"] = np.nan

    return out

def map_wearable_spo2(df):
    # wearable_sports_health_dataset......csv
    out = pd.DataFrame()
    out["spo2_avg"] = pd.to_numeric(df.get("Blood_Oxygen"), errors="coerce")
    out["resting_hr"] = pd.to_numeric(df.get("Heart_Rate"), errors="coerce")
    out["steps"] = pd.to_numeric(df.get("Step_Count"), errors="coerce")
    return out
