# Grafana / Promtail / Loki

## Stack Overview
- **Promtail**: Scrapes Docker container logs, parses structured JSON, adds labels.
- **Loki**: Stores logs and supports LogQL for queries/alerts.
- **Grafana**: Visualizes logs and metrics; provisions dashboards and alerts.

## Structured Logging (from APIs)
- Events: `drift_report`, `prediction`, `prediction_stats`, `monitoring_alert`.
- Example:
```json
{"service": "sentiment-api", "event": "drift_report", "status": "drift_detected", "p_value": 0.01, "threshold": 0.05, "sample_size": 120, "detector_type": "ks"}
```

## Promtail (`deployment/promtail-config.yaml`)
- Docker service discovery via `/var/run/docker.sock`.
- Pipeline:
  - `docker` stage extracts the `log` field.
  - `regex` matches lines containing JSON with `"event"`.
  - `json` parses `service`, `event`, `status`, `level`, `metric`, `detector`, `sample_size`, `p_value`, `threshold`.
  - `labels` exports those fields as labels.
- Key note: container matching uses regex in Grafana/Loki (`container=~".*sentiment-api.*"`).

## Loki
- Exposed at `http://localhost:3100`.
- Receives pushes from Promtail (`/loki/api/v1/push`).

## Grafana
- Exposed at `http://localhost:3000` (admin/admin).
- Datasource provisioning (`deployment/datasources.yaml`):
  - Default Loki datasource (`uid: loki`).
  - Proxy access to `http://loki:3100`.

### Dashboard Provisioning (`deployment/dashboard-provisioning.yaml`)
Folders and files:
- Sentiment API: `grafana-dashboard-sentiment.json`
- Spam API: `grafana-dashboard-spam.json`
- API Models: `dashboard-api-models.json`

### Alerting Provisioning (`deployment/alerting-provisioning.yaml`)
Alert rules (Grafana unified alerting):
- Data Drift Detected - Sentiment
- Low Confidence - Sentiment
- Low Confidence - Spam
- Very Low P-Value - Sentiment
- Multiple Drift Events - Sentiment
- Spam Model Updated
- Sentiment Model Updated

### Dashboard Highlights
**Sentiment**
- Drift detection status
- Mean prediction probability over time
- Average confidence over time
- Drift events and alerts timeline
- P-value distribution (logs/JSON extraction)

**Spam**
- Mean prediction probability over time
- Average confidence over time
- Alerts timeline
- Class ratio and totals

**API Models**
- API request rate
- Model updater status
- Recent model updates

## LogQL Examples
- All drift reports: `{event="drift_report"}`
- Drift detected: `{event="drift_report", status="drift_detected"}`
- Low confidence alerts: `{event="monitoring_alert"} |= "Low confidence prediction detected"`
- Filter by service (regex): `{container=~".*sentiment-api.*"}`
- Prediction rate: `sum(rate({event="prediction"}[1m])) by (service)`
- Drift events over time: `sum(count_over_time({event="drift_report", status="drift_detected"}[5m])) by (service)`
- Average confidence: `avg_over_time({event="prediction_stats"} | json | unwrap avg_confidence [5m])`

## Common Issues
- No data / not valid JSON: ensure regex stage extracts JSON; use regex container match.
- Alerts not firing: verify datasource `uid: loki`, test queries in Explore, check evaluation interval.
- Container name mismatch: always use regex `container=~".*sentiment-api.*"` / `".*spam-api.*"`.

