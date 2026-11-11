"""Validation and cleaning of athlete sensor records."""
import pandas as pd

REQUIRED = ["timestamp","athlete_id","heart_rate","speed",
            "acceleration_x","acceleration_y","acceleration_z"]

def validate_and_clean(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in ["heart_rate","speed","acceleration_x","acceleration_y","acceleration_z"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=REQUIRED).copy()
    out = out.sort_values(["athlete_id","timestamp"]).drop_duplicates(
        ["athlete_id","timestamp"], keep="last"
    ).reset_index(drop=True)
    out["speed"] = out["speed"].clip(lower=0)
    return out
