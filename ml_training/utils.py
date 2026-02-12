import pandas as pd
import numpy as np

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def map_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to standardize to common names:
    age, gender, height_cm, weight_kg, bp_sys, bp_dia, resting_hr, spo2_avg, sleep_hours, steps, screen_time_min
    """
    df = normalize_columns(df)

    ren = {}
    candidates = {
        "age": ["age", "years", "age_years"],
        "gender": ["gender", "sex"],
        "height_cm": ["height", "height_cm", "cm_height"],
        "weight_kg": ["weight", "weight_kg", "kg_weight"],
        "bp_sys": ["bp_sys", "systolic", "sbp", "ap_hi"],
        "bp_dia": ["bp_dia", "diastolic", "dbp", "ap_lo"],
        "resting_hr": ["resting_hr", "heart_rate", "hr", "pulse"],
        "spo2_avg": ["spo2_avg", "spo2", "sp02", "oxygen", "oxygen_saturation"],
        "sleep_hours": ["sleep_hours", "sleep", "sleep_time", "sleepduration"],
        "steps": ["steps", "step_count"],
        "screen_time_min": ["screen_time_min", "screen_time", "screentime"]
    }

    for std, opts in candidates.items():
        for o in opts:
            if o in df.columns:
                ren[o] = std
                break

    df = df.rename(columns=ren)

    # gender normalization to 0/1 if possible
    if "gender" in df.columns:
        g = df["gender"]
        if g.dtype == object:
            df["gender"] = g.astype(str).str.lower().map({
                "m": 1, "male": 1, "1": 1,
                "f": 0, "female": 0, "0": 0
            }).fillna(g)
        df["gender"] = pd.to_numeric(df["gender"], errors="coerce")

    # numeric conversions
    for c in ["age","height_cm","weight_kg","bp_sys","bp_dia","resting_hr","spo2_avg","sleep_hours","steps","screen_time_min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # BMI
    if "height_cm" in df.columns and "weight_kg" in df.columns:
        h_m = df["height_cm"] / 100.0
        df["bmi"] = df["weight_kg"] / (h_m*h_m)
    else:
        df["bmi"] = np.nan

    return df

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return map_common_columns(df)

def safe_concat(dfs):
    dfs = [d for d in dfs if d is not None and len(d) > 0]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
