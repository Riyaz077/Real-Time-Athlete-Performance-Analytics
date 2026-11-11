"""Generate clearly synthetic athlete sensor time-series for demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_demo_data(n_athletes=8, samples_per_athlete=600, seed=42):
    rng = np.random.default_rng(seed)
    frames = []
    start = pd.Timestamp("2026-01-01 08:00:00")
    for athlete_num in range(n_athletes):
        athlete = f"ATH_{athlete_num+1:03d}"
        t = np.arange(samples_per_athlete)
        phase = t / max(samples_per_athlete, 1)
        base_hr = rng.normal(135, 8)
        hr = base_hr + 18*np.sin(2*np.pi*phase) + rng.normal(0, 4, samples_per_athlete)
        speed = np.clip(rng.normal(3.2, .65, samples_per_athlete) + .5*np.sin(2*np.pi*phase), 0, 7)
        ax = rng.normal(0, 1.7, samples_per_athlete)
        ay = rng.normal(0, 1.7, samples_per_athlete)
        az = rng.normal(9.81, 1.2, samples_per_athlete)
        # Add a few synthetic unusual windows; these are not real injury events.
        for _ in range(max(2, samples_per_athlete // 200)):
            idx = rng.integers(20, samples_per_athlete-10)
            hr[idx:idx+5] += rng.uniform(25, 40)
            ax[idx:idx+5] += rng.uniform(8, 12)
            speed[idx:idx+5] *= rng.uniform(.1, .3)
        exertion = np.clip((hr-80)/120, 0, 1) * .55 + np.clip(speed/7,0,1)*.45
        performance = np.clip(100*(.55*speed/7 + .45*(1-np.abs(hr-150)/150)) + rng.normal(0,4,samples_per_athlete),0,100)
        frames.append(pd.DataFrame({
            "timestamp": start + pd.to_timedelta(t, unit="s"),
            "athlete_id": athlete, "heart_rate": np.clip(hr,45,220),
            "speed": speed, "acceleration_x": ax, "acceleration_y": ay,
            "acceleration_z": az, "performance_score": performance,
            "synthetic_data": True
        }))
    return pd.concat(frames, ignore_index=True)

def save_demo_data(path="data/generated/athlete_sensor_demo.csv", **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_demo_data(**kwargs)
    df.to_csv(path, index=False)
    return df
