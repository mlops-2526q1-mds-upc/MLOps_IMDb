"""Periodic downloader to keep the sentiment model updated in a shared volume."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time

import requests

from mlops_imdb.logger import get_logger
from mlops_imdb.modeling.download_production_model import (
    configure_mlflow,
    download_model,
    find_latest_production_eval_run,
    get_experiment_id,
    load_local_metadata,
    load_params,
    save_local_metadata,
)

POLL_SECONDS = int(os.getenv("SENTIMENT_MODEL_POLL_SECONDS", "300"))
API_RELOAD_URL = os.getenv(
    "SENTIMENT_API_RELOAD_URL", "http://sentiment-api:9000/reload"
)  # sentiment-api is the container name in docker compose
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
        logger.info("Notifying sentiment-api to reload model...")
        response = requests.post(API_RELOAD_URL, timeout=10)

        if response.status_code == 200:
            logger.info("sentiment-api reloaded successfully: %s", response.json())
        else:
            logger.error(
                "sentiment-api failed to reload. Status: %s, Response: %s",
                response.status_code,
                response.text,
            )
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not contact sentiment-api at %s. Error: %s", API_RELOAD_URL, exc)


def main():
    target_dir = Path(
        os.getenv("SENTIMENT_MODEL_DIR", "models/sentiment_model_production")
    ).expanduser()
    staging_dir = target_dir.parent / ".sentiment_model_staging"

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
                logger.info("No production-tagged eval run found. Sleeping...")
                time.sleep(POLL_SECONDS)
                continue

            latest_start_ms = run.info.start_time
            if is_up_to_date(target_dir, latest_start_ms):
                logger.info("Local sentiment_model is up to date. Sleeping...")
                time.sleep(POLL_SECONDS)
                continue

            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            download_model(run.info.run_id, str(staging_dir))
            save_local_metadata(staging_dir, run.info.run_id, run.info.start_time)

            if target_dir.exists():
                shutil.rmtree(target_dir)
            staging_dir.rename(target_dir)
            logger.info("Updated sentiment_model from run %s", run.info.run_id)
            notify_api_reload()
        except Exception as exc:
            logger.exception("[sentiment_updater] Error: %s", exc)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
