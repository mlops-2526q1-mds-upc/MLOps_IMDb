"""Download the latest production-tagged sentiment eval pyfunc model from MLflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional

from mlflow.tracking import MlflowClient
import yaml

from mlops_imdb.config import configure_mlflow


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_experiment_id(experiment_name: str) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise RuntimeError(f"Experiment '{experiment_name}' not found in MLflow.")
    return exp.experiment_id


def find_latest_production_eval_run(exp_id: str) -> Optional[object]:
    """Return the most recent run tagged as eval + production."""
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="tags.stage = 'eval'",
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    def is_production_tagged(run_tags: dict) -> bool:
        if run_tags.get("production") is not None:
            return True
        if "labels" in run_tags and "production" in str(run_tags["labels"]).lower():
            return True
        for k, v in run_tags.items():
            if "production" in k.lower():
                return True
        return False

    for run in runs:
        tags = run.data.tags
        if tags.get("stage") != "eval":
            continue
        if is_production_tagged(tags):
            return run
    return None


def download_model(run_id: str, dst_dir: str) -> str:
    client = MlflowClient()
    os.makedirs(dst_dir, exist_ok=True)
    local_path = client.download_artifacts(run_id, "sentiment_model", dst_dir)
    return str(local_path)


def load_local_metadata(dst_dir: Path) -> Optional[dict]:
    meta_path = dst_dir / ".download_meta.json"
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_local_metadata(dst_dir: Path, run_id: str, run_start_ms: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dst_dir / ".download_meta.json"
    downloaded_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_id": run_id,
        "run_start_time_ms": run_start_ms,
        "downloaded_at": downloaded_at,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Download latest production-tagged sentiment eval model artifact from MLflow."
    )
    parser.add_argument(
        "--output-dir",
        default="models/sentiment_model_production",
        help="Local directory where the sentiment_model artifact folder will be downloaded.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional MLflow experiment name to search. Falls back to params.yaml/mlflow config.",
    )
    args = parser.parse_args()

    params = load_params()
    mlflow_cfg = params.get("mlflow", {})
    experiment_name = (
        args.experiment_name
        or mlflow_cfg.get("experiment_name")
        or os.getenv("MLFLOW_EXPERIMENT")
        or "default"
    )
    configure_mlflow(experiment_name)

    exp_id = get_experiment_id(experiment_name)
    run = find_latest_production_eval_run(exp_id)
    if not run:
        raise SystemExit(
            "No production-tagged eval run found. "
            "Ensure the run has tag 'stage=eval' and a 'production' label/tag."
        )

    dst = Path(args.output_dir).expanduser()
    meta = load_local_metadata(dst)
    latest_start = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)

    if meta:
        prev_downloaded_at = meta.get("downloaded_at")
        prev_run_start_ms = meta.get("run_start_time_ms")
        prev_run_id = meta.get("run_id")
        try:
            prev_download_dt = (
                datetime.fromisoformat(prev_downloaded_at) if prev_downloaded_at else None
            )
        except Exception:
            prev_download_dt = None
        try:
            prev_run_start_dt = (
                datetime.fromtimestamp(prev_run_start_ms / 1000, tz=timezone.utc)
                if prev_run_start_ms is not None
                else None
            )
        except Exception:
            prev_run_start_dt = None

        # Only skip download when the upstream run start time matches the stored one.
        # If the upstream start time changes (newer or older), we fetch again because the
        # production tag may have moved.
        if prev_run_start_dt and latest_start == prev_run_start_dt:
            local_ts = (
                prev_download_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                if prev_download_dt
                else "unknown"
            )
            upstream_ts = prev_run_start_dt.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            print(
                f"Local sentiment_model start time matches upstream. "
                f"(local run {prev_run_id}, downloaded at {local_ts}; "
                f"upstream run started {upstream_ts})"
            )
            return

    path = download_model(run.info.run_id, str(dst))
    save_local_metadata(dst, run.info.run_id, run.info.start_time)
    uploaded_ts = latest_start.strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"Downloaded sentiment_model artifact from run {run.info.run_id} "
        f"(run started {uploaded_ts}) to: {path}"
    )


if __name__ == "__main__":
    main()
