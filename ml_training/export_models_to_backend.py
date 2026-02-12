import os
import shutil

SRC = os.path.join("data", "processed")
DST = os.path.join("..", "backend", "ml", "models")

os.makedirs(DST, exist_ok=True)

for f in ["population_baseline.pkl","sleep_model.pkl","spo2_model.pkl","heart_model.pkl","kidney_model.pkl"]:
    src = os.path.join(SRC, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DST, f))
        print("Copied:", f)
    else:
        print("Missing:", f)
