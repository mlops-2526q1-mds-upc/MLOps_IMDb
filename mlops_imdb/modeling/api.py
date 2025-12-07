"""FastAPI endpoint for sentiment prediction using the pyfunc model."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

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


app = FastAPI(title="IMDb Sentiment Analyzer", version="1.0.0")
model = None


def get_model(raise_if_unavailable: bool = True):
    """Return a cached model instance, loading it if needed."""
    global model
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


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Return sentiment probability and label for the provided text."""
    try:
        model_instance = model or get_model()
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
    except Exception as exc:
        logger.exception("Prediction failed for payload")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "ok", "model_uri": DEFAULT_MODEL_URI, "model_loaded": model is not None}
