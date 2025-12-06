import argparse
from pathlib import Path

import mlflow
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict spam probability for input text.")
    parser.add_argument("text", help="Input text to classify.")
    parser.add_argument(
        "--model-uri",
        default="models/spam_model_production/spam_model",
        help="Path or MLflow URI to the saved pyfunc spam model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_uri = args.model_uri

    if not (model_uri.startswith("runs:") or model_uri.startswith("models:")):
        model_uri = str(Path(model_uri).expanduser())

    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.DataFrame({"text": [args.text]})
    proba = model.predict(df)
    # Handle float vs array/Series return types
    if hasattr(proba, "__len__"):
        proba_value = float(proba[0])
    else:
        proba_value = float(proba)
    print(f"Spam probability: {proba_value:.6f}")


if __name__ == "__main__":
    main()
