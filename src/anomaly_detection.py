"""Isolation Forest anomaly detection on sensor-derived features."""
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = ["heart_rate","speed","acceleration_magnitude",
                    "delta_heart_rate","delta_speed","exertion_index"]

def fit_anomaly_detector(df, contamination=0.03, random_state=42):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=random_state)
    model.fit(df[ANOMALY_FEATURES].fillna(0))
    return model

def apply_anomaly_detector(model, df):
    out = df.copy()
    out["anomaly_label"] = model.predict(out[ANOMALY_FEATURES].fillna(0))
    out["is_anomaly"] = out["anomaly_label"] == -1
    out["anomaly_score"] = model.decision_function(out[ANOMALY_FEATURES].fillna(0))
    return out

def save_anomaly_model(model, path="models/anomaly_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model,"features":ANOMALY_FEATURES},path)
