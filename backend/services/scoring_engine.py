# backend/services/scoring.py
import os, joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
MODELS_DIR = os.path.join(BASE_DIR, "models")

def _load(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

MODELS = {
    "heart": _load("heart_model.pkl"),
    "sleep": _load("sleep_model.pkl"),
    "kidney": _load("kidney_model.pkl"),
    "spo2": _load("spo2_model.pkl"),
}

def _risk_from_proba(p_red: float):
    # thresholds: explainable + common triage style
    if p_red >= 0.70: return "Red"
    if p_red >= 0.40: return "Yellow"
    return "Green"

def _model_score(risk):
    return {"Green": 0, "Yellow": 1, "Red": 2}[risk]

# weights with reason:
# Heart & BP = biggest acute risk; SpO2 next; Sleep affects risk but less acute; Kidney early warning but depends on labs
WEIGHTS = {"heart": 0.40, "spo2": 0.25, "sleep": 0.20, "kidney": 0.15}

def score_all(entry: dict):
    # Build feature dict with safe defaults
    x = {
        "bp_sys": float(entry.get("bp_sys") or 0),
        "bp_dia": float(entry.get("bp_dia") or 0),
        "resting_hr": float(entry.get("resting_hr") or 0),
        "spo2_avg": float(entry.get("spo2_avg") or 0),
        "sleep_hours": float(entry.get("sleep_hours") or 0),
        "steps": float(entry.get("steps") or 0),
        "screen_time_min": float(entry.get("screen_time_min") or 0),
        "water_ml": float(entry.get("water_ml") or 0),
        "toilet_freq": float(entry.get("toilet_freq") or 0),
        "alcohol": int(bool(entry.get("alcohol"))),
        "smoking": int(bool(entry.get("smoking"))),
    }

    results = {}
    reasons = []

    # Heart model
    if MODELS["heart"]:
        feats = ["bp_sys","bp_dia","resting_hr","spo2_avg","steps","sleep_hours"]
        X = np.array([[x[f] for f in feats]])
        proba = MODELS["heart"].predict_proba(X)[0]
        p_red = float(proba[-1])  # assumes classes ordered
        risk = _risk_from_proba(p_red)
        results["heart"] = {"risk": risk, "confidence": round(50 + p_red*50, 1)}
        if risk != "Green": reasons.append("Heart/BP pattern elevated")
    else:
        results["heart"] = {"risk": "Unknown", "confidence": 0}

    # Sleep model
    if MODELS["sleep"]:
        feats = ["sleep_hours","spo2_avg","resting_hr","screen_time_min"]
        X = np.array([[x[f] for f in feats]])
        proba = MODELS["sleep"].predict_proba(X)[0]
        p_red = float(proba[-1])
        risk = _risk_from_proba(p_red)
        results["sleep"] = {"risk": risk, "confidence": round(50 + p_red*50, 1)}
        if risk != "Green": reasons.append("Sleep/SpO₂ recovery risk")
    else:
        results["sleep"] = {"risk": "Unknown", "confidence": 0}

    # Kidney model
    if MODELS["kidney"]:
        feats = ["water_ml","toilet_freq","bp_sys","bp_dia"]
        X = np.array([[x[f] for f in feats]])
        proba = MODELS["kidney"].predict_proba(X)[0]
        p_red = float(proba[-1])
        risk = _risk_from_proba(p_red)
        results["kidney"] = {"risk": risk, "confidence": round(50 + p_red*50, 1)}
        if risk != "Green": reasons.append("Hydration/toilet pattern unusual")
    else:
        results["kidney"] = {"risk": "Unknown", "confidence": 0}

    # SpO2 model
    if MODELS["spo2"]:
        feats = ["spo2_avg","sleep_hours","resting_hr"]
        X = np.array([[x[f] for f in feats]])
        proba = MODELS["spo2"].predict_proba(X)[0]
        p_red = float(proba[-1])
        risk = _risk_from_proba(p_red)
        results["spo2"] = {"risk": risk, "confidence": round(50 + p_red*50, 1)}
        if risk != "Green": reasons.append("Oxygen dips risk")
    else:
        results["spo2"] = {"risk": "Unknown", "confidence": 0}

    # Final score (0–100)
    # Convert each risk to points 0/1/2, weighted → 0..2
    total = 0.0
    wsum = 0.0
    for k,w in WEIGHTS.items():
        r = results[k]["risk"]
        if r == "Unknown": 
            continue
        total += _model_score(r) * w
        wsum += w
    norm = (total / (2*wsum)) if wsum > 0 else 0.0
    health_score = int(round(100 * (1 - norm)))  # higher is better

    # Overall label
    if health_score <= 40: overall = "Red"
    elif health_score <= 70: overall = "Yellow"
    else: overall = "Green"

    next_steps = []
    if overall == "Green":
        next_steps = ["Keep logging daily", "Sleep target: 7+ hours", "20–30 min walk today"]
    elif overall == "Yellow":
        next_steps = ["Recheck BP twice daily for 3 days", "Reduce screen time tonight", "If symptoms, consult a doctor"]
    else:
        next_steps = ["Consult a doctor soon for confirmation", "Rest + monitor vitals", "Upload/enter any lab values if available"]

    return {
        "overall": overall,
        "health_score": health_score,
        "components": results,
        "reasons": reasons[:3],
        "next_steps": next_steps[:3],
    }