"""Export trained sentiment model to MLflow pyfunc format for production use."""

import argparse
import json
import os
import shutil
from pathlib import Path

import joblib
import mlflow.pyfunc
import yaml

from mlops_imdb.modeling.mlflow_model import SentimentPyfuncModel


def load_params(path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained sentiment model to MLflow pyfunc format."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to the trained sklearn model (.pkl). Default from params.yaml.",
    )
    parser.add_argument(
        "--vectorizer-path",
        default=None,
        help="Path to the TF-IDF vectorizer (.pkl). Default from params.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/sentiment_model_production",
        help="Directory where to save the MLflow pyfunc model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing production model if it exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params()

    model_path = args.model_path or params["train"]["outputs"]["model_path"]
    vectorizer_path = args.vectorizer_path or params["features"]["outputs"]["vectorizer_path"]
    output_dir = Path(args.output_dir)
    model_output = output_dir / "sentiment_model"

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(vectorizer_path).is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    if model_output.exists():
        if args.force:
            print(f"[save_production_model] Removing existing model at {model_output}")
            shutil.rmtree(model_output)
        else:
            raise FileExistsError(
                f"Output directory already exists: {model_output}. "
                "Use --force to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    preprocess_cfg = params.get("preprocessing", {
        "lowercase": True,
        "remove_html_tags": True,
        "normalize_whitespace": True,
    })

    config = {
        "threshold": args.threshold,
        "preprocess_cfg": preprocess_cfg,
        "model_type": params.get("train", {}).get("model_type", "logistic_regression"),
    }
    config_path = artifacts_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[save_production_model] Saved config -> {config_path}")

    model_artifact_path = artifacts_dir / "model.pkl"
    vectorizer_artifact_path = artifacts_dir / "vectorizer.pkl"
    shutil.copy(model_path, model_artifact_path)
    shutil.copy(vectorizer_path, vectorizer_artifact_path)
    print(f"[save_production_model] Copied model -> {model_artifact_path}")
    print(f"[save_production_model] Copied vectorizer -> {vectorizer_artifact_path}")

    artifacts = {
        "config": config_path.resolve().as_posix(),
        "model": model_artifact_path.resolve().as_posix(),
        "vectorizer": vectorizer_artifact_path.resolve().as_posix(),
    }

    mlflow.pyfunc.save_model(
        path=model_output.resolve().as_posix(),
        python_model=SentimentPyfuncModel(),
        artifacts=artifacts,
    )
    print(f"[save_production_model] Saved MLflow pyfunc model -> {model_output}")
    print("[save_production_model] Done.")


if __name__ == "__main__":
    main()

