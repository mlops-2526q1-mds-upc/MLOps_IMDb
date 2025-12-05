"""MLflow pyfunc wrapper for the IMDb sentiment classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union

import joblib
import mlflow.pyfunc
import pandas as pd

from mlops_imdb.data.prepare import clean_text


class SentimentPyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc model that accepts raw text and returns sentiment probabilities.
    
    Labels: 0 = negative sentiment, 1 = positive sentiment.
    """

    def load_context(self, context):
        """Load model artifacts: config, vectorizer, and classifier."""
        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.vectorizer = joblib.load(context.artifacts["vectorizer"])
        self.model = joblib.load(context.artifacts["model"])
        
        self.preprocess_cfg = self.config.get("preprocess_cfg", {
            "lowercase": True,
            "remove_html_tags": True,
            "normalize_whitespace": True,
        })
        self.threshold = float(self.config.get("threshold", 0.5))

    def _prepare_inputs(self, model_input) -> List[str]:
        """Convert various input types to a list of cleaned text strings."""
        if isinstance(model_input, str):
            texts = [model_input]
        elif isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        elif isinstance(model_input, Iterable):
            texts = [str(x) for x in model_input]
        else:
            texts = [str(model_input)]

        cleaned = [clean_text(t, self.preprocess_cfg) for t in texts]
        return cleaned

    def predict(self, context, model_input) -> Union[float, pd.Series]:
        """Predict sentiment probability for the given text(s).
        
        Returns:
            float or pd.Series: Probability of positive sentiment (label=1).
        """
        texts = self._prepare_inputs(model_input)
        
        # Transform text using TF-IDF vectorizer
        X = self.vectorizer.transform(texts)
        
        # Get probability of positive class
        probas = self.model.predict_proba(X)[:, 1]

        if (
            isinstance(model_input, str)
            or (not hasattr(model_input, "__len__"))
            or len(texts) == 1
        ):
            return float(probas[0])
        return pd.Series(probas, name="prediction")


def save_model(
    model_path: str,
    vectorizer_path: str,
    output_dir: str,
    preprocess_cfg: dict = None,
    threshold: float = 0.5,
) -> str:
    """Save the sentiment model as an MLflow pyfunc model.
    
    Args:
        model_path: Path to the trained sklearn model (.pkl).
        vectorizer_path: Path to the TF-IDF vectorizer (.pkl).
        output_dir: Directory where to save the MLflow model.
        preprocess_cfg: Preprocessing configuration for text cleaning.
        threshold: Classification threshold (default 0.5).
    
    Returns:
        str: Path to the saved MLflow model directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    artifacts_dir = output_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save config
    config = {
        "threshold": threshold,
        "preprocess_cfg": preprocess_cfg or {
            "lowercase": True,
            "remove_html_tags": True,
            "normalize_whitespace": True,
        },
    }
    config_path = artifacts_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    artifacts = {
        "config": str(config_path),
        "vectorizer": str(vectorizer_path),
        "model": str(model_path),
    }
    
    mlflow.pyfunc.save_model(
        path=str(output_path / "sentiment_model"),
        python_model=SentimentPyfuncModel(),
        artifacts=artifacts,
    )
    
    return str(output_path / "sentiment_model")

