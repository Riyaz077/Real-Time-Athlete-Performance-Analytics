# Methodology

## 1. Data ingestion and validation
Load timestamped sensor records. Validate timestamps, athlete identifiers, numeric sensor columns, units, duplicates, and missingness. Sort by athlete and time.

## 2. Preprocessing and feature engineering
Calculate acceleration magnitude as sqrt(ax² + ay² + az²). Generate lagged heart-rate rolling mean and standard deviation, changes in heart rate and speed, acceleration change, normalized sensor values, and an exertion index. Rolling statistics are shifted so they use past observations only.

## 3. Performance prediction
Train a RandomForestRegressor using engineered sensor features and a defined performance_score target. The demo uses a synthetic constructed score for pipeline demonstration. For a real study, define the target using an independently measured, validated performance outcome or a clearly justified proxy. Group-based holdout keeps athlete identities separated between train and test.

## 4. Anomaly detection
Fit IsolationForest on selected sensor features. `-1` denotes an outlier and `1` denotes an inlier. Anomaly scores are statistical flags, not diagnoses. If labeled abnormal events are available, evaluate precision, recall, and F1 on a separate labeled set; otherwise report anomaly rate and review events with domain experts.

## 5. Alert generation
Rule-based alerts flag configured heart-rate and acceleration thresholds, as well as statistical anomalies. Thresholds are configurable and must be calibrated with sports-science experts and device-specific units.

## 6. Evaluation
Performance regression: MAE, RMSE, R². Anomaly detection: precision/recall/F1 only where ground-truth labels exist. Also report latency, throughput, dropped records, and alert counts for a genuine streaming deployment.

## 7. Responsible use
Educational prototype, not a medical or injury prediction system. Use consent, secure data handling, data minimization, expert review, and validated protocols.
