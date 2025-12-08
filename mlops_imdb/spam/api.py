"""FastAPI endpoint for spam prediction using the pyfunc model."""

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

DEFAULT_MODEL_URI = os.getenv("SPAM_MODEL_URI", "models/spam_model_production/spam_model")
logger = get_logger(__name__)


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
        return PredictResponse(probability=proba_val, label=label)
    except Exception as exc:
        logger.exception("Prediction failed for payload")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@public_app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok", "model_uri": DEFAULT_MODEL_URI}


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


# ==========================================
# SERVER ENTRY POINT
# ==========================================
async def serve():
    """Runs both the public and internal servers concurrently."""
    config_public = uvicorn.Config(public_app, host="0.0.0.0", port=8000, log_level="info")
    config_internal = uvicorn.Config(internal_app, host="0.0.0.0", port=9000, log_level="info")

    server_public = uvicorn.Server(config_public)
    server_internal = uvicorn.Server(config_internal)

    # Run both servers at the same time
    await asyncio.gather(server_public.serve(), server_internal.serve())


if __name__ == "__main__":
    # This block runs when you execute `python deployment/spam/main.py`
    asyncio.run(serve())
