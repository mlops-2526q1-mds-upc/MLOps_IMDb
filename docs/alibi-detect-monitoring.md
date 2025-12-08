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

## Automated Hourly Monitoring

### Monitoring Daemons

Both model updater daemons now include automated hourly monitoring checks that trigger monitoring endpoints to collect drift detection, stats, and alerts data.

**Sentiment Model Updater** (`mlops_imdb/modeling/update_model_daemon.py`):
- Triggers every 60 iterations (1 hour when `POLL_SECONDS=60`)
- Calls `/monitoring/drift` - Data drift detection
- Calls `/monitoring/stats` - Prediction statistics
- Calls `/monitoring/alerts` - Recent alerts

**Spam Model Updater** (`mlops_imdb/spam/update_model_daemon.py`):
- Triggers every 60 iterations (1 hour when `POLL_SECONDS=60`)
- Calls `/monitoring/stats` - Prediction statistics
- Calls `/monitoring/alerts` - Recent alerts
- Note: Spam API does not have drift detection

**How it works:**
1. Daemons start automatically when Docker containers start
2. Each iteration increments a counter
3. After 60 iterations (1 hour), monitoring checks are triggered
4. All monitoring events are logged as structured JSON
5. Logs are automatically collected by Promtail and sent to Loki
6. Grafana dashboards visualize the collected metrics

**Configuration:**
- `POLL_SECONDS`: Sleep interval between iterations (default: 60 seconds)
- Counter threshold: 60 iterations = 1 hour
- Monitoring base URLs:
  - Sentiment: `http://sentiment-api:9000`
  - Spam: `http://spam-api:9000`

## Grafana/Loki/Promtail Integration

### Architecture Overview

```
┌─────────────────┐
│  Model Updaters │  (Hourly monitoring triggers)
│  (Daemons)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   API Services  │  (Structured JSON logging)
│  (FastAPI)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Promtail     │  (Log collector/scraper)
│  (Port 9080)    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│      Loki       │  (Log aggregation/storage)
│  (Port 3100)    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Grafana      │  (Visualization & Alerting)
│  (Port 3000)    │
└─────────────────┘
```

### JSON Structured Logging

Monitoring events are logged as JSON for easy parsing by Promtail:

**Drift Report:**
```json
{"service": "sentiment-api", "event": "drift_report", "version": 1, "status": "drift_detected", "p_value": 0.0, "threshold": 0.05, "sample_size": 60, "detector_type": "ks"}
```

**Prediction Event:**
```json
{"service": "sentiment-api", "event": "prediction", "version": 1, "probability": 0.8496803944295178, "label": 1, "sentiment": "positive"}
```

**Monitoring Stats:**
```json
{"service": "sentiment-api", "event": "prediction_stats", "version": 1, "total_predictions": 140, "positive_predictions": 93, "negative_predictions": 47, "positive_ratio": 0.6642857142857143, "avg_confidence": 0.3417424576314452, "confidence_std": 0.21073877118777343, "low_confidence_count": 122, "prediction_drift_detected": false, "alerts": [...]}
```

**Monitoring Alert:**
```json
{"service": "sentiment-api", "event": "monitoring_alert", "version": 1, "level": "info", "message": "Low confidence prediction detected", "metric": "confidence", "value": 0.14475235307351642, "threshold": 0.6, "timestamp": "2025-12-08T19:09:49.355074"}
```

**Event types:**
- `drift_report`: Data drift detection results
- `prediction`: Individual prediction events
- `prediction_stats`: Periodic statistics
- `monitoring_alert`: Alerts (low confidence, ratio drift)

### Promtail Configuration

Promtail (`deployment/promtail-config.yaml`) scrapes Docker container logs and extracts structured JSON fields:

**Key Features:**
- Docker service discovery: Automatically discovers running containers
- JSON parsing: Extracts monitoring fields from structured logs
- Label extraction: Creates labels for filtering in Grafana
- Regex matching: Handles embedded JSON in log messages

**Configuration Details:**

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
        
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'

    pipeline_stages:
      - docker: {}  # Extracts 'log' field from Docker JSON wrapper
      # Match lines containing JSON with "event" field
      - regex:
          expression: '^(?P<json_content>\{.*"event".*\})$'
      # Parse JSON and extract fields
      - json:
          source: json_content
          expressions:
            service: service
            event: event
            status: status
            level: level
            metric: metric
            detector: detector
            sample_size: sample_size
            p_value: p_value
            threshold: threshold
      # Create labels for filtering
      - labels:
          service:
          event:
          status:
          metric:
          detector:
```

**Important Notes:**
- Container names use regex matching: `container=~".*sentiment-api.*"` or `container=~".*spam-api.*"`
- JSON must be on a single line and contain an `"event"` field
- Promtail automatically extracts Docker metadata (container name, stream)

### Loki Configuration

Loki (`deployment/docker-compose.yml`) aggregates and stores logs:

**Configuration:**
- Port: `3100` (exposed to host)
- Storage: In-memory (for development) or persistent volumes (for production)
- Default retention: 24 hours (configurable)

**Access:**
- API: `http://localhost:3100`
- Query API: `http://localhost:3100/loki/api/v1/query`
- Push API: `http://loki:3100/loki/api/v1/push` (internal)

### Grafana Configuration

#### Datasource Provisioning

Grafana automatically connects to Loki via `deployment/datasources.yaml`:

```yaml
apiVersion: 1

datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: true
    editable: true
    uid: loki  # Required for alerting rules
    jsonData:
      maxLines: 1000
```

**Access:** http://localhost:3000 (admin/admin)

#### Dashboard Provisioning

Dashboards are automatically provisioned from JSON files via `deployment/dashboard-provisioning.yaml`:

**Dashboard Folders:**
1. **Sentiment API** (`/etc/grafana/provisioning/dashboards/sentiment/`)
   - `grafana-dashboard-sentiment.json`: Sentiment-specific monitoring

2. **Spam API** (`/etc/grafana/provisioning/dashboards/spam/`)
   - `grafana-dashboard-spam.json`: Spam-specific monitoring

3. **API Models** (`/etc/grafana/provisioning/dashboards/api-models/`)
   - `dashboard-api-models.json`: API health and model updater status

**Dashboard Panels:**

**Sentiment Dashboard:**
- Drift Detection Status (stat panel)
- Mean Prediction Probability Over Time (time series)
- Average Confidence Over Time (time series)
- Monitoring Alerts Timeline (logs panel)
- P-Value Distribution (logs panel with JSON extraction)
- Total Predictions (stat panel)
- Positive/Negative Ratio (stat panel)

**Spam Dashboard:**
- Mean Prediction Probability Over Time (time series)
- Average Confidence Over Time (time series)
- Monitoring Alerts Timeline (logs panel)
- Total Predictions (stat panel)
- Positive/Negative Ratio (stat panel)

**API Models Dashboard:**
- API Request Rate (time series)
- Model Updater Status (stat panels)
- Recent Model Updates (logs panel)

#### Alerting Rules

Alert rules are automatically provisioned via `deployment/alerting-provisioning.yaml`:

**MLOps Alerts Folder:**

1. **Data Drift Detected - Sentiment API**
   - Triggers when drift is detected
   - Severity: Warning
   - Query: `count_over_time({event="drift_report", service=~".*sentiment-api.*", status="drift_detected"}[5m]) > 0`

2. **Low Confidence Prediction - Sentiment API**
   - Triggers on low confidence predictions
   - Severity: Warning
   - Query: `count_over_time({event="monitoring_alert", service=~".*sentiment-api.*"} |= "Low confidence prediction detected" [5m]) > 0`

3. **Low Confidence Prediction - Spam API**
   - Triggers on low confidence predictions
   - Severity: Warning
   - Query: `count_over_time({event="monitoring_alert", service=~".*spam-api.*"} |= "Low confidence prediction detected" [5m]) > 0`

4. **Very Low P-Value (High Drift) - Sentiment API**
   - Triggers on drift with very low p-value
   - Severity: Critical
   - Query: `count_over_time({event="drift_report", service=~".*sentiment-api.*", status="drift_detected"}[5m]) > 0`

5. **Multiple Drift Events - Sentiment API**
   - Triggers on high frequency of drift events
   - Severity: Critical
   - Query: `count_over_time({event="drift_report", service=~".*sentiment-api.*", status="drift_detected"}[10m]) > 3`

**Alerts Folder (Model Updates):**

6. **Spam Model Updated**
   - Triggers when new spam model is deployed
   - Query: `count_over_time({container=~".*spam-model-updater.*"} |= "spam-api reloaded successfully" [1m])`

7. **Sentiment Model Updated**
   - Triggers when new sentiment model is deployed
   - Query: `count_over_time({container=~".*sentiment-model-updater.*"} |= "sentiment-api reloaded successfully" [1m])`

### Grafana LogQL Queries

**Example queries for exploring data:**

```logql
# All drift reports
{event="drift_report"}

# Drift detected events only
{event="drift_report", status="drift_detected"}

# All monitoring alerts
{event="monitoring_alert"}

# Filter by service (using regex for container names)
{container=~".*sentiment-api.*"}

# Low confidence alerts
{event="monitoring_alert"} |= "Low confidence prediction detected"

# Prediction events with high probability
{event="prediction"} | json | probability > 0.8

# Stats events
{event="prediction_stats"}

# Count drift events over time
count_over_time({event="drift_report", status="drift_detected"}[5m])

# Rate of predictions
rate({event="prediction"}[1m])

# Average confidence from stats
avg_over_time({event="prediction_stats"} | json | unwrap avg_confidence [5m])
```

**Time Series Queries:**

```logql
# Prediction rate by service
sum(rate({event="prediction"}[1m])) by (service)

# Drift events over time
sum(count_over_time({event="drift_report", status="drift_detected"}[5m])) by (service)

# Average confidence over time
avg_over_time({event="prediction_stats"} | json | unwrap avg_confidence [5m])
```

## Docker Configuration

### Docker Compose Services

The `deployment/docker-compose.yml` orchestrates all services:

**Application Services:**
- `spam-api`: Spam prediction API (ports 8000, 9000)
- `sentiment-api`: Sentiment prediction API (ports 8001, 9001)
- `spam-model-updater`: Model updater daemon with hourly monitoring
- `sentiment-model-updater`: Model updater daemon with hourly monitoring
- `ui`: Streamlit UI (port 8501)

**Monitoring Stack:**
- `loki`: Log aggregation (port 3100)
- `promtail`: Log collector (internal)
- `grafana`: Visualization and alerting (port 3000)

### Volume Mounts

**Sentiment API** requires access to training artifacts for drift detection:

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

**Spam API** also mounts models and data for consistency:

```yaml
spam-api:
  volumes:
    - spam-models:/shared/models
    - ../models:/app/models:ro
    - ../data/processed:/app/data/processed:ro
  ports:
    - "8000:8000"   # Public API
    - "9000:9000"   # Internal API (monitoring)
```

**Grafana** mounts provisioning files:

```yaml
grafana:
  volumes:
    - ./datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml
    - ./dashboard-provisioning.yaml:/etc/grafana/provisioning/dashboards/dashboard-provisioning.yaml
    - ./alerting-provisioning.yaml:/etc/grafana/provisioning/alerting/alerting-provisioning.yaml
    - ./grafana-dashboard-sentiment.json:/etc/grafana/provisioning/dashboards/sentiment/grafana-dashboard-sentiment.json
    - ./grafana-dashboard-spam.json:/etc/grafana/provisioning/dashboards/spam/grafana-dashboard-spam.json
    - ./dashboard-api-models.json:/etc/grafana/provisioning/dashboards/api-models/dashboard-api-models.json
```

**Promtail** mounts Docker socket for service discovery:

```yaml
promtail:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./promtail-config.yaml:/etc/promtail/config.yml
```

### Network Configuration

```yaml
networks:
  sentiment-net:    # Sentiment API and updater
  spam-net:         # Spam API and updater
  monitoring:       # Loki, Promtail, Grafana (shared)
```

All services are connected to appropriate networks for isolation and communication.

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

### 5. Grafana Dashboard "No Data" or "Not Valid JSON"

**Problem:** Grafana dashboards showed "no data" or "not valid json" errors.

**Causes:**
- Container name patterns didn't match Docker Compose naming (e.g., `deployment-sentiment-api-1`)
- LogQL queries used exact matches instead of regex
- JSON parsing failed due to embedded JSON in log messages

**Solutions:**
- Updated all queries to use regex: `container=~".*sentiment-api.*"` or `service=~".*sentiment-api.*"`
- Modified Promtail config to use regex stage before JSON parsing to extract embedded JSON
- Corrected LogQL syntax for extracting numeric values: `| json | unwrap p_value` or `| json | p_value`
- When `unwrap` wasn't supported, used Logs panels with automatic JSON field extraction

### 6. Monitoring Daemon Not Triggering

**Problem:** Monitoring checks weren't being triggered by daemons.

**Cause:** 
- Monitoring check code was placed before model update logic
- `continue` statements skipped the `time.sleep()` at the end
- Loop ran continuously, executing monitoring checks every iteration instead of hourly

**Solution:**
- Moved monitoring check and counter increment to after model update logic
- Removed `continue` statements and restructured code so `time.sleep()` always runs
- Added logging to show count progress: `"Count: %d/60 - Not triggering hourly monitoring checks yet"`
- Ensured counter increments and sleep happen on every iteration

## Testing

### Quick Test Commands

**Health Checks:**
```powershell
# Sentiment API health
Invoke-RestMethod -Uri "http://localhost:8001/health"

# Spam API health
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

**Making Predictions:**
```powershell
# Generate predictions for monitoring (sentiment)
1..60 | ForEach-Object { 
    $null = Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method Post `
        -ContentType "application/json" `
        -Body (@{text="This movie was absolutely fantastic!"} | ConvertTo-Json -Compress)
}

# Generate predictions for monitoring (spam)
1..60 | ForEach-Object { 
    $null = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post `
        -ContentType "application/json" `
        -Body (@{text="Free money now! Click here!"} | ConvertTo-Json -Compress)
}
```

**Monitoring Endpoints:**
```powershell
# Check drift (sentiment only)
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/drift"

# Check stats (sentiment)
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/stats"

# Check stats (spam)
Invoke-RestMethod -Uri "http://localhost:9000/monitoring/stats"

# Check alerts (sentiment)
Invoke-RestMethod -Uri "http://localhost:9001/monitoring/alerts"

# Check alerts (spam)
Invoke-RestMethod -Uri "http://localhost:9000/monitoring/alerts"
```

**Viewing Logs:**
```powershell
# API logs
docker logs deployment-sentiment-api-1 --tail 50
docker logs deployment-spam-api-1 --tail 50

# Daemon logs (monitoring triggers)
docker logs deployment-sentiment-model-updater-1 --tail 50
docker logs deployment-spam-model-updater-1 --tail 50

# All logs via docker-compose
cd deployment
docker-compose logs --tail=50 sentiment-api spam-api
docker-compose logs --tail=50 sentiment-model-updater spam-model-updater
```

**Testing Monitoring Daemons:**
```powershell
# Watch daemon logs for count progress
docker logs -f deployment-sentiment-model-updater-1

# After 60 iterations (1 hour), you should see:
# "Triggering hourly sentiment monitoring checks..."
# "Triggered hourly sentiment monitoring checks"
```

**Grafana Access:**
- URL: http://localhost:3000
- Default credentials: admin/admin
- Dashboards: Navigate to "Dashboards" → Browse folders
  - Sentiment API
  - Spam API
  - API Models
- Alerts: Navigate to "Alerting" → Alert rules

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

## Deployment

### Starting the Stack

```powershell
cd deployment
docker-compose up -d --build
```

### Stopping the Stack

```powershell
docker-compose down
```

### Rebuilding After Code Changes

```powershell
docker-compose down
docker-compose build
docker-compose up -d
```

### Viewing Service Status

```powershell
docker-compose ps
```

### Accessing Services

- **Sentiment API**: http://localhost:8001 (public), http://localhost:9001 (internal)
- **Spam API**: http://localhost:8000 (public), http://localhost:9000 (internal)
- **Streamlit UI**: http://localhost:8501
- **Grafana**: http://localhost:3000
- **Loki**: http://localhost:3100

## Monitoring Workflow

1. **Startup**: All services start automatically via Docker Compose
2. **Initialization**: APIs load models and initialize monitoring components
3. **Prediction Collection**: Each prediction is recorded in monitoring buffers
4. **Hourly Checks**: Model updater daemons trigger monitoring endpoints every hour
5. **Logging**: All monitoring events are logged as structured JSON
6. **Collection**: Promtail scrapes logs from Docker containers
7. **Storage**: Loki aggregates and stores logs
8. **Visualization**: Grafana dashboards display metrics and trends
9. **Alerting**: Grafana alerts trigger on drift detection or low confidence

## Best Practices

1. **Monitor Regularly**: Ensure hourly monitoring checks are running
2. **Check Dashboards**: Review Grafana dashboards daily for trends
3. **Respond to Alerts**: Investigate drift and low confidence alerts promptly
4. **Data Quality**: Monitor prediction distributions for unexpected shifts
5. **Model Updates**: Track model deployment events via alerts
6. **Log Retention**: Configure Loki retention based on storage capacity
7. **Container Names**: Always use regex patterns for container name matching

## Troubleshooting

### Monitoring Not Triggering

- Check daemon logs: `docker logs deployment-sentiment-model-updater-1`
- Verify counter is incrementing: Look for "Count: X/60" messages
- Ensure `POLL_SECONDS=60` in environment

### No Data in Grafana

- Verify Promtail is running: `docker-compose ps promtail`
- Check Loki is receiving logs: `docker-compose logs loki`
- Verify container name patterns use regex: `container=~".*sentiment-api.*"`
- Check JSON logs are being parsed: View raw logs in Grafana Explore

### Alerts Not Firing

- Verify alert rules are provisioned: Check Grafana Alerting → Alert rules
- Check datasource UID matches: Should be `loki` in all queries
- Verify LogQL syntax: Test queries in Grafana Explore first
- Check alert evaluation interval: Default is 30s-1m

## References

- [Alibi-Detect Documentation](https://docs.seldon.io/projects/alibi-detect/)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Promtail Documentation](https://grafana.com/docs/loki/latest/clients/promtail/)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)

