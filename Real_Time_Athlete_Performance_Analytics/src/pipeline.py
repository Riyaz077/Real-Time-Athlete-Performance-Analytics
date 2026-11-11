"""Run reproducible demo pipeline from command line."""
import argparse
from pathlib import Path
import pandas as pd
from src.data_generator import save_demo_data
from src.preprocessing import validate_and_clean
from src.feature_engineering import engineer_features
from src.modeling import train_performance_model, save_model
from src.anomaly_detection import fit_anomaly_detector, apply_anomaly_detector, save_anomaly_model
from src.alerts import generate_alerts

def run(input_path="data/generated/athlete_sensor_demo.csv", generate_demo=False):
    if generate_demo or not Path(input_path).exists():
        save_demo_data(input_path)
    raw = pd.read_csv(input_path)
    clean = validate_and_clean(raw)
    featured = engineer_features(clean)
    model, metrics, test_idx = train_performance_model(featured)
    save_model(model)
    detector = fit_anomaly_detector(featured)
    save_anomaly_model(detector)
    scored = featured.copy()
    from src.modeling import FEATURES
    scored["predicted_performance_score"] = model.predict(scored[FEATURES])
    scored = apply_anomaly_detector(detector, scored)
    alerts = generate_alerts(scored)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([metrics]).assign(Dataset="Athlete Sensor Demo",Model="Random Forest").to_csv("results/performance_metrics.csv",index=False)
    scored.to_csv("results/athlete_analytics_output.csv",index=False)
    alerts.to_csv("results/alert_records.csv",index=False)
    pd.DataFrame([{"records":len(scored),"athletes":scored.athlete_id.nunique(),
                   "anomalies":int(scored.is_anomaly.sum()),
                   "anomaly_rate":float(scored.is_anomaly.mean()),
                   "alerts":len(alerts)}]).to_csv("results/anomaly_summary.csv",index=False)
    print("Pipeline complete. Metrics:",metrics)
    print("Outputs saved under results/ and trained artifacts under models/.")
    print("DEMO DATA IS SYNTHETIC — do not interpret as real athlete validation.")
    return scored, alerts, metrics

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--generate-demo",action="store_true")
    parser.add_argument("--input",default="data/generated/athlete_sensor_demo.csv")
    args=parser.parse_args()
    run(args.input,args.generate_demo)
