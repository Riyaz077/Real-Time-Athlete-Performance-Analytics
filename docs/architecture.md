# System architecture

```mermaid
flowchart TD
 A[Athlete sensor stream / CSV] --> B[Validation and preprocessing]
 B --> C[Feature engineering]
 C --> D[Random Forest performance estimation]
 C --> E[Isolation Forest anomaly detection]
 D --> F[Rule-based alert generation]
 E --> F
 F --> G[Evaluation and saved outputs]
 G --> H[Streamlit dashboard]
```

The current code simulates sequential sensor records and processes a CSV batch. It is a real-time analytics prototype, not a production streaming infrastructure such as Kafka or MQTT. Update this diagram if the implementation changes.
