"""FastAPI endpoint for sentiment prediction using the pyfunc model."""

from __future__ import annotations

import asyncio
import os
import json
import logging
from pathlib import Path
import json
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from mlops_imdb.logger import get_logger
from mlops_imdb.monitoring import DriftDetector, DriftReport, PredictionMonitor
from mlops_imdb.monitoring.utils import (
    get_monitoring_config,
    initialize_sentiment_monitoring,
)

DEFAULT_MODEL_URI = os.getenv(
    "SENTIMENT_MODEL_URI", "models/sentiment_model_production/sentiment_model"
)
logger = get_logger(__name__)
SERVICE_NAME = "sentiment-api"
LOG_VERSION = 1
_monitor_logger: Optional[logging.Logger] = None

# Monitoring components (initialized lazily)
drift_detector: Optional[DriftDetector] = None
prediction_monitor: Optional[PredictionMonitor] = None
monitoring_enabled: bool = True


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze for sentiment.")


class PredictResponse(BaseModel):
    probability: float = Field(..., description="Positive sentiment probability between 0 and 1.")
    label: int = Field(..., description="Predicted label (1=positive, 0=negative).")
    sentiment: str = Field(..., description="Human-readable sentiment (positive/negative).")


def _resolve_model_uri(uri: str) -> str:
    """Resolve model URI to a valid path or MLflow URI."""
    if uri.startswith("runs:") or uri.startswith("models:"):
        return uri
    resolved = Path(uri).expanduser().resolve()
    return resolved.as_posix()


def load_model(uri: str):
    """Load the MLflow pyfunc model from the given URI."""
    resolved = _resolve_model_uri(uri)
    try:
        model = mlflow.pyfunc.load_model(resolved)
        logger.info("Loaded sentiment model from %s", resolved)
        return model
    except Exception as exc:
        logger.exception("Failed to load sentiment model from %s", resolved)
        raise RuntimeError(f"Failed to load model from {resolved}") from exc



public_app = FastAPI(title="IMDb Sentiment Analyzer", version="1.0.0")
model = load_model(DEFAULT_MODEL_URI)
model_reload_lock = Lock()


def initialize_monitoring():
    """Initialize drift detection and prediction monitoring components."""
    global drift_detector, prediction_monitor, monitoring_enabled

    try:
        config = get_monitoring_config()
        monitoring_enabled = config.get("enabled", True)

        if not monitoring_enabled:
            logger.info("Monitoring disabled in configuration")
            return

        drift_detector, prediction_monitor = initialize_sentiment_monitoring(
            sample_size=config.get("drift_detection", {}).get("sample_size", 1000),
            p_threshold=config.get("drift_detection", {}).get("p_threshold", 0.05),
        )

        if drift_detector is not None:
            logger.info("Drift detector initialized successfully")
        else:
            logger.warning("Drift detector could not be initialized")

        if prediction_monitor is not None:
            logger.info("Prediction monitor initialized successfully")
        else:
            logger.warning("Prediction monitor could not be initialized")

    except Exception as exc:
        logger.warning("Failed to initialize monitoring: %s", exc)
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

try:
    initialize_monitoring()
except Exception as exc:
    logger.warning("Monitoring initialization failed: %s", exc)


def get_model(raise_if_unavailable: bool = True):
    """Return a cached model instance, loading it if needed."""
    global model
    if model is not None:
        return model
    with model_reload_lock:
        if model is not None:
            return model
        try:
            model = load_model(DEFAULT_MODEL_URI)
            return model
        except Exception as exc:
            if raise_if_unavailable:
                raise HTTPException(status_code=503, detail="Model not available") from exc
            logger.warning(
                "Sentiment model could not be loaded from %s; API will return 503 until available",
                _resolve_model_uri(DEFAULT_MODEL_URI),
            )
            return None


@public_app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Return sentiment probability and label for the provided text."""
    try:
        model_instance = get_model()
        df = pd.DataFrame({"text": [payload.text]})
        proba = model_instance.predict(df)
        proba_val = float(proba[0]) if hasattr(proba, "__len__") else float(proba)
        label = int(proba_val >= 0.5)
        sentiment = "positive" if label == 1 else "negative"
        logger.info(
            "Predicted sentiment for input text: '%s' has sentiment: %s with probability: %f",
            payload.text,
            sentiment,
            proba_val,
        )

        # Log prediction as structured JSON for monitoring
        _log_monitoring_event(
            event="prediction",
            payload={
                "probability": proba_val,
                "label": label,
                "sentiment": sentiment,
            },
            level="info",
        )

        if monitoring_enabled:
            try:
                if drift_detector is not None:
                    drift_detector.add_sample(payload.text)
                if prediction_monitor is not None:
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

        return PredictResponse(probability=proba_val, label=label, sentiment=sentiment)
    except HTTPException:
        raise
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
        "model_loaded": model is not None,
        "monitoring_enabled": monitoring_enabled,
        "drift_detector_active": drift_detector is not None,
        "prediction_monitor_active": prediction_monitor is not None,
    }


internal_app = FastAPI(title="Sentiment Internal Admin", version="1.0.0")


@internal_app.post("/reload")
def reload_model():
    """Restricted endpoint to reload the model from disk."""
    logger.info("Internal request received: Reloading sentiment model...")
    global model
    with model_reload_lock:
        try:
            new_model = load_model(DEFAULT_MODEL_URI)
            model = new_model
            logger.info("Sentiment model reloaded from %s", DEFAULT_MODEL_URI)
            return {"status": "reloaded", "uri": DEFAULT_MODEL_URI}
        except Exception as exc:
            logger.exception("Failed to reload sentiment model")
            raise HTTPException(status_code=500, detail="Failed to reload model") from exc


@internal_app.get("/monitoring/drift")
def check_drift():
    """Check for data drift in recent predictions."""
    if not monitoring_enabled or drift_detector is None:
        return {
            "status": "disabled",
            "message": "Drift detection not enabled or not initialized",
        }

    try:
        report = drift_detector.detect_drift(use_buffer=True)
        _log_monitoring_event(
            event="drift_report",
            payload={
                "status": report.status.value,
                "p_value": report.p_value,
                "threshold": report.threshold,
                "sample_size": report.sample_size,
                "detector_type": drift_detector.detector_type,
            },
        )
        return report.to_dict()
    except Exception as exc:
        logger.exception("Drift check failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    logger.info("Resetting monitoring state...")
    try:
        if drift_detector is not None:
            drift_detector.clear_buffer()
        if prediction_monitor is not None:
            prediction_monitor.reset()
        return {"status": "reset", "message": "Monitoring state cleared"}
    except Exception as exc:
        logger.exception("Failed to reset monitoring")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@internal_app.post("/monitoring/reinitialize")
def reinitialize_monitoring():
    """Reinitialize monitoring components from training data."""
    logger.info("Reinitializing monitoring...")
    try:
        initialize_monitoring()
        return {
            "status": "reinitialized",
            "drift_detector_active": drift_detector is not None,
            "prediction_monitor_active": prediction_monitor is not None,
        }
    except Exception as exc:
        logger.exception("Failed to reinitialize monitoring")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def serve():
    """Runs both the public and internal servers concurrently."""
    config_public = uvicorn.Config(public_app, host="0.0.0.0", port=8000, log_level="info")
    config_internal = uvicorn.Config(internal_app, host="0.0.0.0", port=9000, log_level="info")

    server_public = uvicorn.Server(config_public)
    server_internal = uvicorn.Server(config_internal)

    await asyncio.gather(server_public.serve(), server_internal.serve())


if __name__ == "__main__":
    asyncio.run(serve())
