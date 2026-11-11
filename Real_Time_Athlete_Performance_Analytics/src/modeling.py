"""Random Forest performance score regression."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["heart_rate","speed","acceleration_magnitude","heart_rate_rolling_mean",
            "heart_rate_rolling_std","delta_heart_rate","delta_speed",
            "acceleration_change","heart_rate_norm","speed_norm",
            "acceleration_norm","exertion_index"]

def train_performance_model(df, target="performance_score", random_state=42):
    if target not in df.columns:
        raise ValueError(f"Training target '{target}' is missing. Provide a validated target or proxy.")
    data = df.dropna(subset=FEATURES+[target]).copy()
    # Group split prevents rows from the same athlete appearing in both sets.
    groups = data["athlete_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data[FEATURES], data[target], groups))
    X_train, X_test = data.iloc[train_idx][FEATURES], data.iloc[test_idx][FEATURES]
    y_train, y_test = data.iloc[train_idx][target], data.iloc[test_idx][target]
    model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                  random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"MAE": mean_absolute_error(y_test,pred),
               "RMSE": mean_squared_error(y_test,pred)**0.5,
               "R2": r2_score(y_test,pred)}
    return model, metrics, test_idx

def save_model(model, path="models/performance_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model,"features":FEATURES},path)
