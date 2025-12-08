# Alibi-Detect Monitoring Integration

This document describes the implementation of drift detection and model monitoring using the `alibi-detect` library, integrated with the Grafana/Loki/Promtail logging stack.

## Overview

Alibi-detect provides algorithms for outlier detection, adversarial detection, and drift detection. In this project, we use it to:

- **Data Drift Detection**: Detect when input data distributions shift from the training distribution
- **Output Drift Detection**: Monitor prediction distributions for anomalies
- **Alerting**: Generate alerts for low confidence predictions and ratio drift

## Architecture

```
+------------------+     +------------------+     +------------------+
|  Sentiment API   |     |    Spam API      |     |    Streamlit UI  |
|  (port 8001)     |     |   (port 8000)    |     |   (port 8501)    |
+--------+---------+     +--------+---------+     +------------------+
         |                        |
         v                        v
+------------------+     +------------------+
| Drift Detector   |     | Prediction       |
| (KS Test on      |     | Monitor          |
|  TF-IDF features)|     | (Output drift)   |
+--------+---------+     +--------+---------+
         |                        |
         +------------+-----------+
                      |
                      v
              +-------+-------+
              | JSON Logs     |
              | (structured)  |
              +-------+-------+
                      |
                      v
              +-------+-------+
              |   Promtail    |
              | (log scraper) |
              +-------+-------+
                      |
                      v
              +-------+-------+
              |     Loki      |
              | (log storage) |
              +-------+-------+
                      |
                      v
              +-------+-------+
              |   Grafana     |
              | (visualization)|
              +---------------+
```

## Implementation Details

### 1. Sentiment API Monitoring

The sentiment API (`mlops_imdb/modeling/api.py`) includes full drift detection:

**Components:**
- `DriftDetector`: Uses Kolmogorov-Smirnov (KS) test on TF-IDF features
- `PredictionMonitor`: Tracks prediction distributions and confidence scores

**How it works:**
1. On startup, loads the TF-IDF vectorizer and training data from mounted volumes
2. Initializes a KS drift detector with reference features from training data
3. On each prediction, adds the input text to a sliding window buffer
4. Transforms text to TF-IDF features for drift comparison
5. When drift check is requested, compares current buffer against reference distribution

**Configuration:**
- Reference data: 1000 samples from training set
- P-value threshold: 0.05 (drift detected if p < 0.05)
- Minimum samples: 50 (required before drift detection runs)
- Window size: 500 samples (sliding window)

### 2. Spam API Monitoring

The spam API (`mlops_imdb/spam/api.py`) uses prediction monitoring only:

**Components:**
- `PredictionMonitor`: Tracks prediction distributions

**Why no drift detector?**
The spam model uses token embeddings (not TF-IDF), so we only monitor output distributions rather than input features.

**Tracked metrics:**
- Prediction counts (spam vs not spam)
- Confidence scores
- Low confidence prediction counts
- Alerts for ratio drift

### 3. Monitoring Module Structure

```
mlops_imdb/monitoring/
    __init__.py              # Public exports
    drift_detector.py        # DriftDetector class (alibi-detect KS/Chi2/Tabular)
    prediction_monitor.py    # PredictionMonitor class (output monitoring)
    utils.py                 # Initialization helpers, config loading
```

## API Endpoints

### Public API (port 8000/8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Make prediction (also records to monitoring buffer) |
| `/health` | GET | Health check including monitoring status |

### Internal API (port 9000/9001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/monitoring/drift` | GET | Check data drift status (sentiment only) |
| `/monitoring/stats` | GET | Get prediction statistics |
| `/monitoring/alerts` | GET | Get recent monitoring alerts |
| `/monitoring/reset` | POST | Reset monitoring state |
| `/monitoring/reinitialize` | POST | Reinitialize monitoring from training data |
| `/reload` | POST | Reload the ML model |

### Example Responses

**Drift Check (`/monitoring/drift`):**
```json
{
  "status": "drift_detected",
  "p_value": 0.0,
  "threshold": 0.05,
  "drift_score": 0.85,
  "drift_detected": true,
  "sample_size": 60,
  "detector_type": "ks"
}
```

**Stats (`/monitoring/stats`):**
```json
{
  "total_predictions": 100,
  "positive_predictions": 65,
  "negative_predictions": 35,
  "positive_ratio": 0.65,
  "avg_confidence": 0.72,
  "confidence_std": 0.15,
  "low_confidence_count": 8,
  "prediction_drift_detected": false,
  "alerts": []
}
```

## Grafana/Loki/Promtail Integration

### JSON Structured Logging

Monitoring events are logged as JSON for easy parsing by Promtail:

```json
{"service": "sentiment-api", "event": "drift_report", "version": 1, "status": "drift_detected", "p_value": 0.0, "threshold": 0.05, "sample_size": 60, "detector_type": "ks"}
```

**Event types:**
- `drift_report`: Data drift detection results
- `monitoring_alert`: Alerts (low confidence, ratio drift)
- `prediction_stats`: Periodic statistics

### Promtail Configuration

The `deployment/promtail-config.yaml` includes JSON parsing:

```yaml
pipeline_stages:
  - docker: {}
  - json:
      expressions:
        service: service
        event: event
        status: status
        level: level
        metric: metric
        detector: detector
  - labels:
      service:
      event:
      status:
      metric:
      detector:
```

### Grafana Queries

Access Grafana at http://localhost:3000 (admin/admin).

**Example Loki queries:**

```logql
# All drift reports
{event="drift_report"}

# Drift detected events only
{event="drift_report", status="drift_detected"}

# All monitoring alerts
{event="monitoring_alert"}

# Filter by service
{service="sentiment-api"}

# Critical alerts only
{event="monitoring_alert"} |= "critical"
```

## Docker Configuration

### Volume Mounts

The sentiment API requires access to training artifacts for drift detection:

```yaml
sentiment-api:
  volumes:
    - sentiment-models:/shared/models
    - ../models:/app/models:ro          # TF-IDF vectorizer
    - ../data/processed:/app/data/processed:ro  # Training data
  ports:
    - "8001:8000"   # Public API
    - "9001:9000"   # Internal API (monitoring)
```

### Network Configuration

```yaml
networks:
  sentiment-net:    # Sentiment API network
  spam-net:         # Spam API network
  monitoring:       # Loki/Promtail/Grafana network
```

## Difficulties Encountered and Solutions

### 1. Drift Detector Not Initializing

**Problem:** Sentiment API showed `drift_detector_active: False` after deployment.

**Cause:** The TF-IDF vectorizer (`models/tfidf_vectorizer.pkl`) and training data (`data/processed/imdb_train_clean.csv`) were not available inside the Docker container.

**Solution:** Added bind mounts in `docker-compose.yml`:
```yaml
volumes:
  - ../models:/app/models:ro
  - ../data/processed:/app/data/processed:ro
```

### 2. Slow/Blocking Requests

**Problem:** Prediction requests were taking a long time, especially after many predictions.

**Cause:** 
- `get_stats()` was calling `check_output_drift()` internally, running KS test on every stats call
- Deadlock potential: both methods tried to acquire the same lock

**Solution:** 
- Removed automatic drift check from `get_stats()` 
- Made drift checking a separate explicit operation via `/monitoring/drift`
- Stats endpoint is now lightweight and instant

### 3. PowerShell Sequential Requests

**Problem:** Batch prediction loops appeared to hang.

**Cause:** PowerShell `ForEach-Object` runs requests sequentially, waiting for each to complete.

**Solution:** Use output suppression for faster iteration:
```powershell
1..60 | ForEach-Object { $null = Invoke-RestMethod ... }
```

### 4. Monitoring State Reset

**Problem:** After container restart, all monitoring data was lost.

**Cause:** Monitoring state is stored in-memory (intentional for lightweight operation).

**Solution:** This is by design. For production persistence, implement save/load methods (available in `DriftDetector.save()` / `DriftDetector.load()`).

## Testing

### Quick Test Commands

```powershell
# Check health
Invoke-RestMethod -Uri "http://localhost:8001/health"

# Make predictions
1..60 | ForEach-Object { $null = Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post -ContentType "application/json" -Body '{"text": "Test movie review"}' }

# Check drift
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/drift"

# Check stats
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/stats"

# View logs
docker logs deployment-sentiment-api-1 --tail 50
```

### Expected Drift Detection

Drift will be detected when:
- Input text is significantly different from training data (movie reviews)
- P-value falls below 0.05 threshold
- At least 50 samples in the buffer

Example: Sending generic text like "Movie sample 1, 2, 3..." will trigger drift because it differs from actual movie review language.

## Configuration

All monitoring parameters are configurable in `params.yaml`:

```yaml
monitoring:
  enabled: true
  drift_detection:
    enabled: true
    p_threshold: 0.05
    detector_type: ks  # Options: ks, chi2, tabular
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

## Dependencies

Added to `pyproject.toml`:
```toml
dependencies = [
    "alibi-detect>=0.12.0",
    # ... other dependencies
]
```

## References

- [Alibi-Detect Documentation](https://docs.seldon.io/projects/alibi-detect/)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)

