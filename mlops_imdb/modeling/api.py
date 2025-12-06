"""FastAPI endpoint for sentiment prediction using the pyfunc model."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

DEFAULT_MODEL_URI = os.getenv(
    "SENTIMENT_MODEL_URI", "models/sentiment_model_production/sentiment_model"
)


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
        return mlflow.pyfunc.load_model(resolved)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {resolved}") from exc


app = FastAPI(title="IMDb Sentiment Analyzer", version="1.0.0")
model = load_model(DEFAULT_MODEL_URI)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """Return sentiment probability and label for the provided text."""
    try:
        df = pd.DataFrame({"text": [payload.text]})
        proba = model.predict(df)
        proba_val = float(proba[0]) if hasattr(proba, "__len__") else float(proba)
        label = int(proba_val >= 0.5)
        sentiment = "positive" if label == 1 else "negative"
        return PredictResponse(probability=proba_val, label=label, sentiment=sentiment)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_uri": DEFAULT_MODEL_URI}
