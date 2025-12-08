# Docker / Compose

## Services (deployment/docker-compose.yml)
- `sentiment-api` (8001 public, 9001 internal)
- `spam-api` (8000 public, 9000 internal)
- `sentiment-model-updater` (daemon + hourly monitoring)
- `spam-model-updater` (daemon + hourly monitoring)
- `ui` (Streamlit, 8501)
- `loki` (3100)
- `promtail`
- `grafana` (3000)

## Volumes
- Models: `sentiment-models:/shared/models`, `spam-models:/shared/models`
- Bind mounts for artifacts:
  - `../models:/app/models:ro`
  - `../data/processed:/app/data/processed:ro`
- Grafana provisioning:
  - `./datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml`
  - `./dashboard-provisioning.yaml:/etc/grafana/provisioning/dashboards/dashboard-provisioning.yaml`
  - `./alerting-provisioning.yaml:/etc/grafana/provisioning/alerting/alerting-provisioning.yaml`
  - Dashboards JSON in sentiment/spam/api-models folders
- Promtail: `/var/run/docker.sock` + `./promtail-config.yaml`

## Networks
- `sentiment-net` (sentiment API + updater + UI)
- `spam-net` (spam API + updater + UI)
- `monitoring` (Grafana, Loki, Promtail)

## Commands
```powershell
# Start / rebuild
cd deployment
docker-compose up -d --build

# Stop
docker-compose down

# Rebuild only app image
docker-compose build

# Status
docker-compose ps
```

## Access URLs
- Sentiment API: http://localhost:8001 (public), http://localhost:9001 (internal)
- Spam API: http://localhost:8000 (public), http://localhost:9000 (internal)
- Streamlit UI: http://localhost:8501
- Grafana: http://localhost:3000
- Loki: http://localhost:3100

