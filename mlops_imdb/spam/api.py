"""FastAPI endpoint for spam prediction using the pyfunc model."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from threading import Lock
from typing import Optional
import json
import logging

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from mlops_imdb.logger import get_logger
from mlops_imdb.monitoring import PredictionMonitor
from mlops_imdb.monitoring.utils import get_monitoring_config, initialize_spam_monitoring

DEFAULT_MODEL_URI = os.getenv("SPAM_MODEL_URI", "models/spam_model_production/spam_model")
logger = get_logger(__name__)
SERVICE_NAME = "spam-api"
LOG_VERSION = 1
_monitor_logger: Optional[logging.Logger] = None

# Monitoring components
prediction_monitor: Optional[PredictionMonitor] = None
monitoring_enabled: bool = True


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify as spam or not spam.")


class PredictResponse(BaseModel):
    probability: float = Field(..., description="Spam probability between 0 and 1.")
    label: int = Field(..., description="Predicted label (1=spam, 0=not spam).")


def _resolve_model_uri(uri: str) -> str:
    if uri.startswith("runs:") or uri.startswith("models:"):
        return uri
    return str(Path(uri).expanduser())


def load_model(uri: str):
    resolved = _resolve_model_uri(uri)
    try:
        model = mlflow.pyfunc.load_model(resolved)
        logger.info("Loaded spam model from %s", resolved)
        return model
    except Exception as exc:
        logger.exception("Failed to load model from %s", resolved)
        raise RuntimeError(f"Failed to load model from {resolved}") from exc


# ==========================================
# APP 1: PUBLIC API (Port 8000)
# ==========================================

public_app = FastAPI(title="Spam Predictor", version="1.0.0")
model = load_model(DEFAULT_MODEL_URI)
model_reload_lock = Lock()


def initialize_monitoring():
    """Initialize prediction monitoring components."""
    global prediction_monitor, monitoring_enabled

    try:
        config = get_monitoring_config()
        monitoring_enabled = config.get("enabled", True)

        if not monitoring_enabled:
            logger.info("Monitoring disabled in configuration")
            return

        _, prediction_monitor = initialize_spam_monitoring(
            p_threshold=config.get("drift_detection", {}).get("p_threshold", 0.05),
        )

        if prediction_monitor is not None:
            logger.info("Spam prediction monitor initialized successfully")
        else:
            logger.warning("Spam prediction monitor could not be initialized")

    except Exception as exc:
        logger.warning("Failed to initialize spam monitoring: %s", exc)
        monitoring_enabled = False


def _get_monitor_logger() -> logging.Logger:
    """Return a logger configured for JSON monitoring output."""
    global _monitor_logger
    if _monitor_logger is None:
        mon_logger = logging.getLogger(f"{SERVICE_NAME}.monitoring")
        mon_logger.propagate = False
        mon_logger.setLevel(logging.INFO)
        if not mon_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            mon_logger.addHandler(handler)
        _monitor_logger = mon_logger
    return _monitor_logger


def _log_monitoring_event(event: str, payload: dict, level: str = "info") -> None:
    """Emit a structured JSON log for monitoring events."""
    mon_logger = _get_monitor_logger()
    log_body = {
        "service": SERVICE_NAME,
        "event": event,
        "version": LOG_VERSION,
        **payload,
    }
    try:
        message = json.dumps(log_body)
        if level == "warning":
            mon_logger.warning(message)
        else:
            mon_logger.info(message)
    except Exception as exc:
        logger.debug("Failed to emit monitoring log: %s", exc)


# Initialize monitoring on startup
try:
    initialize_monitoring()
except Exception as exc:
    logger.warning("Spam monitoring initialization failed: %s", exc)


@public_app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Return spam probability and label for the provided text."""
    try:
        df = pd.DataFrame({"text": [payload.text]})
        proba = model.predict(df)
        proba_val = float(proba[0]) if hasattr(proba, "__len__") else float(proba)
        label = int(proba_val >= 0.5)
        logger.info(
            "Predicted spam for input text: '%s' was classified as %d with probability: %f",
            payload.text,
            label,
            proba_val,
        )

        # Log prediction as structured JSON for monitoring
        _log_monitoring_event(
            event="prediction",
            payload={
                "probability": proba_val,
                "label": label,
            },
            level="info",
        )

        # Record prediction for monitoring (non-blocking)
        if monitoring_enabled and prediction_monitor is not None:
            try:
                alert = prediction_monitor.record_prediction(proba_val, label)
                if alert is not None:
                    _log_monitoring_event(
                        event="monitoring_alert",
                        payload={
                            "level": alert.level.value,
                            "message": alert.message,
                            "metric": alert.metric,
                            "value": alert.value,
                            "threshold": alert.threshold,
                        },
                        level="warning",
                    )
            except Exception as mon_exc:
                logger.debug("Monitoring failed (non-critical): %s", mon_exc)

        return PredictResponse(probability=proba_val, label=label)
    except Exception as exc:
        logger.exception("Prediction failed for payload")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@public_app.get("/health")
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "model_uri": DEFAULT_MODEL_URI,
        "monitoring_enabled": monitoring_enabled,
        "prediction_monitor_active": prediction_monitor is not None,
    }


# ==========================================
# APP 2: INTERNAL API (Port 9000)
# ==========================================
internal_app = FastAPI(title="Spam Internal Admin", version="1.0.0")


@internal_app.post("/reload")
def reload_model():
    """Restricted endpoint to reload the model from disk."""
    logger.info("Internal request received: Reloading model...")
    global model
    with model_reload_lock:
        try:
            new_model = load_model(DEFAULT_MODEL_URI)
            model = new_model
            logger.info("Model reloaded from %s", DEFAULT_MODEL_URI)
            return {"status": "reloaded", "uri": DEFAULT_MODEL_URI}
        except Exception as exc:
            logger.exception("Failed to reload model")
            raise HTTPException(status_code=500, detail="Failed to reload model") from exc


@internal_app.get("/monitoring/stats")
def get_monitoring_stats():
    """Get prediction monitoring statistics."""
    if not monitoring_enabled or prediction_monitor is None:
        return {
            "status": "disabled",
            "message": "Prediction monitoring not enabled or not initialized",
        }

    try:
        stats = prediction_monitor.get_stats()
        _log_monitoring_event(
            event="prediction_stats",
            payload=stats.to_dict(),
        )
        return stats.to_dict()
    except Exception as exc:
        logger.exception("Failed to get monitoring stats")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@internal_app.get("/monitoring/alerts")
def get_monitoring_alerts(limit: int = 50):
    """Get recent monitoring alerts."""
    if not monitoring_enabled or prediction_monitor is None:
        return {"alerts": [], "message": "Monitoring not enabled"}

    try:
        alerts = prediction_monitor.get_alerts(limit=limit)
        return {"alerts": [a.to_dict() for a in alerts]}
    except Exception as exc:
        logger.exception("Failed to get alerts")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@internal_app.post("/monitoring/reset")
def reset_monitoring():
    """Reset monitoring state and clear buffers."""
    logger.info("Resetting spam monitoring state...")
    try:
        if prediction_monitor is not None:
            prediction_monitor.reset()
        return {"status": "reset", "message": "Monitoring state cleared"}
    except Exception as exc:
        logger.exception("Failed to reset monitoring")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@internal_app.post("/monitoring/reinitialize")
def reinitialize_monitoring():
    """Reinitialize monitoring components."""
    logger.info("Reinitializing spam monitoring...")
    try:
        initialize_monitoring()
        return {
            "status": "reinitialized",
            "prediction_monitor_active": prediction_monitor is not None,
        }
    except Exception as exc:
        logger.exception("Failed to reinitialize monitoring")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==========================================
# SERVER ENTRY POINT
# ==========================================
async def serve():
    """Runs both the public and internal servers concurrently."""
    config_public = uvicorn.Config(public_app, host="0.0.0.0", port=8000, log_level="info")
    config_internal = uvicorn.Config(internal_app, host="0.0.0.0", port=9000, log_level="info")

    server_public = uvicorn.Server(config_public)
    server_internal = uvicorn.Server(config_internal)

    await asyncio.gather(server_public.serve(), server_internal.serve())


if __name__ == "__main__":
    asyncio.run(serve())
