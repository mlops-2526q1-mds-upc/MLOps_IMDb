"""Periodic downloader to keep spam model updated in a shared volume."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

import requests  # <--- NEW IMPORT

from mlops_imdb.logger import get_logger
from mlops_imdb.spam.download_production_model import (
    configure_mlflow,
    download_model,
    find_latest_production_eval_run,
    get_experiment_id,
    load_local_metadata,
    load_params,
    save_local_metadata,
)

POLL_SECONDS = int(os.getenv("SPAM_MODEL_POLL_SECONDS", "60"))
# The internal URL. Note the port 9000 (internal only).
API_RELOAD_URL = os.getenv(
    "SPAM_API_RELOAD_URL", "http://spam-api:9000/reload"
)  # the spam-api is the container name in the docker compose
SPAM_MONITORING_BASE_URL = os.getenv("SPAM_MONITORING_BASE_URL", "http://spam-api:9000")

logger = get_logger(__name__)


def is_up_to_date(target_dir: Path, latest_run_start_ms: int) -> bool:
    meta = load_local_metadata(target_dir)
    if not meta:
        return False
    prev_run_start_ms = meta.get("run_start_time_ms")
    if prev_run_start_ms is None:
        return False
    # Only consider up-to-date when the stored run has the same start timestamp.
    # If the production tag moves (newer or older), we need to refresh.
    return latest_run_start_ms == prev_run_start_ms


def notify_api_reload():
    """
    Sends a request to the API's internal admin port to reload the model from disk.
    This ensures the API serves the new files immediately.
    """
    try:
        logger.info("Notifying spam-api to reload model...")
        # We use a timeout so the updater doesn't hang if the API is frozen
        response = requests.post(API_RELOAD_URL, timeout=10)

        if response.status_code == 200:
            logger.info("spam-api reloaded successfully: %s", response.json())
        else:
            logger.error(
                "spam-api failed to reload. Status: %s, Response: %s",
                response.status_code,
                response.text,
            )
    except requests.exceptions.RequestException as e:
        # We log a warning but don't crash, so the updater keeps running
        logger.warning("Could not contact spam-api at %s. Error: %s", API_RELOAD_URL, e)


def trigger_monitoring_checks():
    """
    Trigger hourly monitoring checks: stats and alerts.
    Called every hour (60 minutes) to ensure monitoring data is collected.
    Note: Spam API does not have drift detection.
    """
    try:
        logger.info("Triggering hourly spam monitoring checks...")

        # Get monitoring stats
        try:
            response = requests.get(f"{SPAM_MONITORING_BASE_URL}/monitoring/stats", timeout=30)
            if response.status_code == 200:
                stats = response.json()
                logger.info(
                    "Monitoring stats retrieved: %d total predictions",
                    stats.get("total_predictions", 0),
                )
            else:
                logger.warning("Stats check returned status %s", response.status_code)
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to get stats: %s", exc)

        # Get monitoring alerts
        try:
            response = requests.get(f"{SPAM_MONITORING_BASE_URL}/monitoring/alerts", timeout=30)
            if response.status_code == 200:
                alerts = response.json()
                logger.info("Retrieved %d monitoring alerts", len(alerts.get("alerts", [])))
            else:
                logger.warning("Alerts check returned status %s", response.status_code)
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to get alerts: %s", exc)

    except Exception as exc:
        logger.exception("Error during monitoring checks: %s", exc)


def main():
    target_dir = Path(os.getenv("SPAM_MODEL_DIR", "models/spam_model_production")).expanduser()
    staging_dir = target_dir.parent / ".spam_model_staging"

    count = 0
    while True:
        try:
            params = load_params()
            mlflow_cfg = params.get("mlflow", {})
            experiment_name = (
                mlflow_cfg.get("experiment_name") or os.getenv("MLFLOW_EXPERIMENT") or "default"
            )
            configure_mlflow(experiment_name)

            exp_id = get_experiment_id(experiment_name)
            run = find_latest_production_eval_run(exp_id)
            if not run:
                logger.info("No production-tagged spam_eval run found. Sleeping...")
            else:
                latest_start_ms = run.info.start_time
                if not is_up_to_date(target_dir, latest_start_ms):
                    # Clean staging and download there
                    if staging_dir.exists():
                        shutil.rmtree(staging_dir)
                    staging_dir.mkdir(parents=True, exist_ok=True)
                    download_model(run.info.run_id, str(staging_dir))
                    save_local_metadata(staging_dir, run.info.run_id, run.info.start_time)

                    # Atomic swap: remove old target and move staging into place
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    staging_dir.rename(target_dir)
                    logger.info("Updated spam_model from run %s", run.info.run_id)
                    notify_api_reload()
                else:
                    logger.info("Local spam_model is up to date. Sleeping...")

        except Exception as exc:
            logger.exception("[updater] Error: %s", exc)

        # Increment counter and trigger hourly monitoring checks
        count += 1
        if count >= 5:  # 60 minutes = 1 hour (when sleeping every 1 minute)
            count = 0
            trigger_monitoring_checks()
            logger.info("Triggered hourly spam monitoring checks")
        else:
            logger.info("Count: %d/60 - Not triggering hourly spam monitoring checks yet", count)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
