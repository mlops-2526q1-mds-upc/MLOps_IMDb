"""CLI tool for predicting sentiment on input text."""

import argparse
from pathlib import Path

import mlflow
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict sentiment (positive/negative) for input text."
    )
    parser.add_argument("text", help="Input text to classify.")
    parser.add_argument(
        "--model-uri",
        default="models/sentiment_model_production/sentiment_model",
        help="Path or MLflow URI to the saved pyfunc sentiment model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_uri = args.model_uri

    if not (model_uri.startswith("runs:") or model_uri.startswith("models:")):
        resolved = Path(model_uri).expanduser().resolve()
        model_uri = resolved.as_posix()

    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.DataFrame({"text": [args.text]})
    proba = model.predict(df)
    
    if hasattr(proba, "__len__"):
        proba_value = float(proba[0])
    else:
        proba_value = float(proba)
    
    label = int(proba_value >= 0.5)
    sentiment = "positive" if label == 1 else "negative"
    
    print(f"Sentiment: {sentiment}")
    print(f"Probability (positive): {proba_value:.6f}")
    print(f"Label: {label}")


if __name__ == "__main__":
    main()

