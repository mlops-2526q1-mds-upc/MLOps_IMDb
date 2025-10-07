from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from loguru import logger

# --- Load .env early from project root, regardless of CWD ---
_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path)
logger.info(f"Loaded .env from: {_env_path or '<not found>'}")

# --- Paths ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for p in [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

# Integrate loguru with tqdm (optional)
try:
    from tqdm import tqdm  # noqa: F401

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# --- MLflow / DagsHub settings from env (NO runs/tags here) ---
MLFLOW_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME: Optional[str] = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD: Optional[str] = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "default")

DAGSHUB_REPO: Optional[str] = os.getenv("DAGSHUB_REPO")  # "<USER>/<REPO>" (optional)
ENV_NAME: str = os.getenv("ENV", "local")


def configure_mlflow(experiment_name: Optional[str] = None) -> None:
    """
    Configure MLflow using values loaded from .env via this config module.
    This function ONLY sets tracking URI and experiment.
    It does NOT start a run or set tags.
    """
    import mlflow

    if not MLFLOW_TRACKING_URI:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Put it in your .env at project root, "
            "e.g. https://dagshub.com/<USER>/<REPO>.mlflow"
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or MLFLOW_EXPERIMENT)

    print("[MLflow] configured — tracking_uri:", mlflow.get_tracking_uri())
