# Alibi-Detect Monitoring

## What It Covers
- Drift detection for the sentiment model (KS/Chi2/Tabular via `alibi-detect`)
- Prediction monitoring for both sentiment and spam (confidence, class ratios, alerts)
- Internal monitoring endpoints (ports 9001 sentiment, 9000 spam)

## Components
- `DriftDetector` (`mlops_imdb/monitoring/drift_detector.py`): TF-IDF + KS by default; supports Chi2/Tabular. Sliding window, min samples, save/load helpers.
- `PredictionMonitor` (`mlops_imdb/monitoring/prediction_monitor.py`): Tracks prediction counts, confidence stats, ratio drift, low-confidence alerts.
- Helpers (`mlops_imdb/monitoring/utils.py`): Initialization from `params.yaml`.

## API Integration
### Sentiment (`mlops_imdb/modeling/api.py`)
- Uses `DriftDetector` and `PredictionMonitor`.
- Logs structured JSON events (`prediction`, `drift_report`, `prediction_stats`, `monitoring_alert`).
- Internal endpoints:
  - `GET /monitoring/drift`
  - `GET /monitoring/stats`
  - `GET /monitoring/alerts`
  - `POST /monitoring/reset`
  - `POST /monitoring/reinitialize`
  - `POST /reload` (model reload)

### Spam (`mlops_imdb/spam/api.py`)
- Uses `PredictionMonitor` only (no input drift detector).
- Internal endpoints:
  - `GET /monitoring/stats`
  - `GET /monitoring/alerts`
  - `POST /monitoring/reset`
  - `POST /monitoring/reinitialize`
  - `POST /reload`

## Configuration (`params.yaml`)
```yaml
monitoring:
  enabled: true
  drift_detection:
    enabled: true
    p_threshold: 0.05
    detector_type: ks  # ks | chi2 | tabular
    min_samples: 50
    window_size: 500
    sample_size: 1000
  prediction_monitoring:
    enabled: true
    window_size: 1000
    confidence_threshold: 0.6
    positive_ratio_warning: 0.3
    positive_ratio_critical: 0.2
  alerting:
    enabled: true
    log_alerts: true
```

## Hourly Monitoring Triggers
- Sentiment updater: calls `/monitoring/drift`, `/monitoring/stats`, `/monitoring/alerts` every 60 iterations (POLL_SECONDS=60) in `mlops_imdb/modeling/update_model_daemon.py`.
- Spam updater: calls `/monitoring/stats`, `/monitoring/alerts` every 60 iterations in `mlops_imdb/spam/update_model_daemon.py`.
- Both daemons log progress (`Count: X/60`) and run `time.sleep()` every loop to ensure hourly cadence.

## Testing (PowerShell)
```powershell
# Health
Invoke-RestMethod -Uri "http://localhost:8001/health"
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Generate predictions (sentiment)
1..60 | ForEach-Object {
  $null = Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post `
    -ContentType "application/json" `
    -Body (@{text="This movie was absolutely fantastic!"} | ConvertTo-Json -Compress)
}

# Generate predictions (spam)
1..60 | ForEach-Object {
  $null = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post `
    -ContentType "application/json" `
    -Body (@{text="Free money now! Click here!"} | ConvertTo-Json -Compress)
}

# Monitoring checks
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/drift"
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/stats"
Invoke-RestMethod -Uri "http://localhost:9000/monitoring/stats"
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/alerts"
Invoke-RestMethod -Uri "http://localhost:9000/monitoring/alerts"
```

## Structured Log Examples
- Drift report:
```json
{"service": "sentiment-api", "event": "drift_report", "status": "drift_detected", "p_value": 0.0, "threshold": 0.05, "sample_size": 60, "detector_type": "ks"}
```
- Prediction:
```json
{"service": "sentiment-api", "event": "prediction", "probability": 0.85, "label": 1, "sentiment": "positive"}
```
- Stats:
```json
{"service": "sentiment-api", "event": "prediction_stats", "total_predictions": 140, "avg_confidence": 0.34, "low_confidence_count": 122, "alerts": [...]}
```
- Alert:
```json
{"service": "sentiment-api", "event": "monitoring_alert", "message": "Low confidence prediction detected", "metric": "confidence", "threshold": 0.6}
```

