"""FastAPI endpoint for sentiment prediction using the pyfunc model."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field
import uvicorn

from mlops_imdb.logger import get_logger

DEFAULT_MODEL_URI = os.getenv(
    "SENTIMENT_MODEL_URI", "models/sentiment_model_production/sentiment_model"
)
logger = get_logger(__name__)


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


# ==========================================
# APP 1: PUBLIC API (Port 8000)
# ==========================================

public_app = FastAPI(title="IMDb Sentiment Analyzer", version="1.0.0")
model = load_model(DEFAULT_MODEL_URI)
model_reload_lock = Lock()


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
            "Predicted sentiment for input text: %s has sentiment: %s with probability: %f",
            payload.text,
            sentiment,
            proba_val,
        )
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
    return {"status": "ok", "model_uri": DEFAULT_MODEL_URI, "model_loaded": model is not None}


# ==========================================
# APP 2: INTERNAL API (Port 9000)
# ==========================================
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
