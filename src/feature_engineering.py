"""Create causal rolling and sensor-derived features."""
import numpy as np
import pandas as pd

def engineer_features(df, window=10):
    out = df.copy().sort_values(["athlete_id","timestamp"])
    out["acceleration_magnitude"] = np.sqrt(
        out["acceleration_x"]**2 + out["acceleration_y"]**2 + out["acceleration_z"]**2
    )
    grouped = out.groupby("athlete_id", group_keys=False)
    out["heart_rate_rolling_mean"] = grouped["heart_rate"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    out["heart_rate_rolling_std"] = grouped["heart_rate"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=2).std()
    ).fillna(0)
    out["delta_heart_rate"] = grouped["heart_rate"].diff().fillna(0)
    out["delta_speed"] = grouped["speed"].diff().fillna(0)
    out["acceleration_change"] = grouped["acceleration_magnitude"].diff().fillna(0)
    # Transparent normalized sensor features; clipping avoids extreme scale dominance.
    out["heart_rate_norm"] = ((out["heart_rate"] - 60) / 160).clip(0, 1.5)
    out["speed_norm"] = (out["speed"] / 8).clip(0, 1.5)
    out["acceleration_norm"] = (out["acceleration_magnitude"] / 20).clip(0, 1.5)
    out["exertion_index"] = 0.5*out["heart_rate_norm"] + 0.3*out["speed_norm"] + 0.2*out["acceleration_norm"]
    return out
