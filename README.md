# Real-Time Athlete Performance Analytics and Anomaly Detection

An end-to-end athlete sensor analytics project demonstrating data preprocessing, feature engineering, performance-score estimation, anomaly detection, rule-based alerts, and an interactive Streamlit dashboard.

> **Data note:** The repository includes a reproducible synthetic sensor-data generator for demonstration. Synthetic data is not real athlete evidence and must not be presented as measured results from real athletes. Replace it with authorized, consented sensor data before making real-world claims.

## Project workflow

```text
Athlete sensor data (CSV / simulated stream)
        ↓
Validation, cleaning, timestamp ordering
        ↓
Feature engineering
(acceleration magnitude, rolling HR, speed/acceleration changes,
normalized sensor features, exertion index)
        ↓
Random Forest performance-score regression
        ↓
Isolation Forest anomaly detection
        ↓
Rule-based alert generation
        ↓
Evaluation, visualizations, saved outputs
        ↓
Streamlit analytics dashboard
```

## Key capabilities

- Sensor-data ingestion and validation
- Batch and sequential streaming simulation
- Athlete-level and time-window features
- Random Forest regression for a defined performance-score target
- Isolation Forest for unusual sensor-pattern detection
- Rule-based alerts with severity and explanation
- Evaluation metrics, charts, CSV exports, and dashboard

## Repository layout

```text
.
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── config/settings.yaml
├── data/README.md
├── notebooks/01_Athlete_Performance_Analytics.ipynb
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── anomaly_detection.py
│   ├── alerts.py
│   └── pipeline.py
├── models/README.md
├── results/README.md
├── screenshots/README.md
├── docs/
│   ├── methodology.md
│   ├── architecture.md
│   ├── System_Architecture.png
│   ├── Project_Report.pdf
│   └── Methodology.pdf
├── presentation/project_presentation.pptx
└── tests/test_pipeline.py
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --generate-demo
streamlit run app.py
```

The demo command generates synthetic data under `data/generated/` and pipeline outputs under `results/`. Run it before opening the dashboard.

## Input schema

CSV columns:
- `timestamp`
- `athlete_id`
- `heart_rate`
- `speed`
- `acceleration_x`, `acceleration_y`, `acceleration_z`
- `performance_score` (optional supervised training target; 0–100)

If using real sensor data, document device type, sampling frequency, units, missingness, athlete consent, and data provenance. The performance score in this demo is a constructed target; replace it with a validated outcome or clearly define a defensible proxy.

## Modeling approach

- **Performance estimation:** RandomForestRegressor trained on engineered sensor features to estimate `performance_score`.
- **Anomaly detection:** IsolationForest trained on sensor features to flag unusual patterns. An anomaly is a statistical outlier, not a medical diagnosis or proof of injury.
- **Alerts:** Transparent rule-based thresholds on sensor signals and anomaly flags. Tune thresholds with domain experts; do not use as medical advice.

## Evaluation

Regression: MAE, RMSE, R².  
Anomaly detection: where ground-truth labels exist, precision, recall, F1, and confusion matrix. Without labels, report anomaly rate and review flagged events manually. Do not treat synthetic labels as real-world validation.

## Responsible use

This is an educational analytics prototype, not a clinical or athlete-safety system. Avoid making injury, diagnosis, or fitness-to-play decisions from model output. Use informed consent, secure storage, data minimization, expert review, and validated thresholds.

## Author

Mohamed Riyaz S. | VIT Chennai


