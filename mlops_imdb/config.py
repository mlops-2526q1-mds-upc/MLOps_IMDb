from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------
# 1️⃣ Load .env from project root
# ---------------------------------------------------------------------

env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
    logger.info(f"Loaded .env from: {env_path}")
else:
    logger.warning(".env file not found — environment variables will be read from system only")

# ---------------------------------------------------------------------
# 2️⃣ Paths
# ---------------------------------------------------------------------

# Path to the project root (folder containing mlops_imdb/)
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

for p in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 3️⃣ Environment variables (from .env or system)
# ---------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "default")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
ENV = os.getenv("ENV", "local")

# ---------------------------------------------------------------------
# 4️⃣ Helper: configure MLflow
# ---------------------------------------------------------------------


def configure_mlflow(experiment_name: str | None = None) -> None:
    """Configure MLflow tracking and experiment using loaded environment variables."""

    if not MLFLOW_TRACKING_URI:
        logger.error("⚠️  MLFLOW_TRACKING_URI not set — using local file store (./mlruns)")
        raise RuntimeError("MLFLOW_TRACKING_URI is required for MLflow configuration")
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_to_use = experiment_name or MLFLOW_EXPERIMENT

    client = MlflowClient()
    existing = client.get_experiment_by_name(experiment_to_use)
    if existing and existing.lifecycle_stage != "active":
        client.restore_experiment(existing.experiment_id)

    mlflow.set_experiment(experiment_to_use)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


# ---------------------------------------------------------------------
# 5️⃣ Debug print (optional)
# ---------------------------------------------------------------------


def print_env_summary() -> None:
    """Print which variables were loaded (without exposing secrets)."""
    print("🔧 Environment summary:")
    print(f"  PROJ_ROOT: {PROJ_ROOT}")
    print(f"  MLFLOW_TRACKING_URI: {bool(MLFLOW_TRACKING_URI)}")
    print(f"  MLFLOW_TRACKING_USERNAME: {MLFLOW_TRACKING_USERNAME}")
    print(f"  MLFLOW_TRACKING_PASSWORD: {'✅ set' if MLFLOW_TRACKING_PASSWORD else '❌ missing'}")
    print(f"  MLFLOW_EXPERIMENT: {MLFLOW_EXPERIMENT}")
    print(f"  DAGSHUB_REPO: {DAGSHUB_REPO}")
    print(f"  ENV: {ENV}")


# ---------------------------------------------------------------------
# Run directly for verification
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print_env_summary()
    configure_mlflow()
