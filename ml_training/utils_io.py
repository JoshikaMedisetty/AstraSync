import os
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

def find_file(possible_names):
    """
    Finds a file in data/raw by exact name OR wildcard.
    Example: ["wearable_sports_health_dataset*.csv"]
    """
    for name in possible_names:
        path = os.path.join(RAW_DIR, name)
        matches = glob.glob(path)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"None found in {RAW_DIR}: {possible_names}")

def read_csv_any(possible_names):
    path = find_file(possible_names)
    return pd.read_csv(path), path
