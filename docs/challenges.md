# Challenges and Solutions

## 1) Drift Detector Not Initializing
- **Issue:** `drift_detector_active: False` (sentiment).
- **Cause:** TF-IDF vectorizer and training data not mounted in container.
- **Fix:** Bind mount `../models` and `../data/processed` into sentiment-api; restart.

## 2) Slow / Blocking Requests
- **Issue:** Stats calls were very slow.
- **Cause:** `get_stats()` invoked `check_output_drift()` (KS) and both contended on the same lock.
- **Fix:** Separate drift check into `/monitoring/drift`; make stats lightweight.

## 4) Monitoring State Reset on Restart
- **Issue:** Monitoring data lost after restart.
- **Cause:** In-memory by design.
- **Fix:** Accept for lightweight mode; use `DriftDetector.save/load` for persistence if needed.

## 5) Grafana “No Data” / “Not Valid JSON”
- **Issues:** Empty panels or JSON errors.
- **Causes:** Container name mismatch, strict equals instead of regex, JSON not parsed.
- **Fixes:** Use regex (`container=~".*sentiment-api.*"`), add regex stage before JSON in Promtail, correct LogQL (`json | unwrap field`), fallback to Logs panels where needed.

## 6) Monitoring Daemon Not Triggering Hourly
- **Issue:** Monitoring checks ran every loop or never slept.
- **Cause:** Monitoring code before model logic + `continue` skipping `sleep`.
- **Fix:** Move counter/monitoring after model logic; always sleep; add progress logs (`Count: X/60`).

## 7) Docker Build / Runtime Issues
- **Missing image:** Build locally (`docker-compose build`).
- **Docker Desktop off:** Start Docker Desktop then `docker-compose up -d --build`.
- **Mount path errors:** Ensure provisioning and data files exist at mounted paths.

## 8) Alerts Not Firing
- **Issue:** Grafana alerts inactive.
- **Causes:** Wrong datasource UID, LogQL errors, missing regex filters.
- **Fix:** Use `uid: loki`, test queries in Explore, regex container match, correct evaluation windows.

## Quick Troubleshooting Checklist
- Check Promtail: `docker-compose ps promtail`
- Check Loki logs: `docker-compose logs loki`
- Test queries in Grafana Explore with regex container filters
- Verify monitoring daemons log `Count: X/60` and trigger messages
- Rebuild/restart: `docker-compose down && docker-compose up -d --build`

