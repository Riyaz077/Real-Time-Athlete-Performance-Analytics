# Data

The demo pipeline creates `data/generated/athlete_sensor_demo.csv` automatically. This is synthetic data for code demonstration only.

For real data, place an authorized CSV under `data/real/` and use the same schema:
`timestamp, athlete_id, heart_rate, speed, acceleration_x, acceleration_y, acceleration_z, performance_score`

Never upload identifiable or sensitive athlete data without appropriate consent and authorization. The `.gitignore` excludes `data/real/`.
