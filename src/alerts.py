"""Transparent rule-based alert generation; thresholds are configurable."""
import pandas as pd

def generate_alerts(df, high_hr=190, high_accel=12, anomaly_only=False):
    out = df.copy()
    conditions = [
        out["heart_rate"] >= high_hr,
        out["acceleration_magnitude"] >= high_accel,
        out.get("is_anomaly", pd.Series(False,index=out.index)).astype(bool)
    ]
    labels = ["HIGH_HEART_RATE","HIGH_ACCELERATION","STATISTICAL_ANOMALY"]
    out["alert_types"] = [
        ", ".join(label for cond,label in zip(conditions,labels) if bool(cond.loc[idx]))
        for idx in out.index
    ]
    out["alert_severity"] = out["alert_types"].apply(
        lambda s: "High" if "HIGH_HEART_RATE" in s or "HIGH_ACCELERATION" in s
        else ("Review" if s else "None")
    )
    alerts = out[out["alert_types"] != ""].copy()
    if anomaly_only:
        alerts = alerts[alerts["is_anomaly"]]
    return alerts
