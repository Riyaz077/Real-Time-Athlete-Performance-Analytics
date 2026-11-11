from src.data_generator import generate_demo_data
from src.preprocessing import validate_and_clean
from src.feature_engineering import engineer_features
from src.anomaly_detection import fit_anomaly_detector, apply_anomaly_detector
from src.alerts import generate_alerts

def test_demo_pipeline_smoke():
    raw=generate_demo_data(n_athletes=3,samples_per_athlete=80)
    clean=validate_and_clean(raw)
    features=engineer_features(clean)
    assert len(features)==len(clean)
    model=fit_anomaly_detector(features)
    scored=apply_anomaly_detector(model,features)
    assert "is_anomaly" in scored
    alerts=generate_alerts(scored)
    assert set(alerts.columns).issuperset({"alert_types","alert_severity"})
