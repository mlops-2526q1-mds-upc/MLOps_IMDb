"""Export trained sentiment model to MLflow pyfunc format for production use."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

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

    model_path = Path(args.model_path or params["train"]["outputs"]["model_path"])
    vectorizer_path = Path(
        args.vectorizer_path or params["features"]["outputs"]["vectorizer_path"]
    )
    output_dir = Path(args.output_dir)
    model_output = output_dir / "sentiment_model"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.is_file():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    if model_output.exists():
        if args.force:
            print(f"[save_production_model] Removing existing model at {model_output}")
            shutil.rmtree(model_output)
        else:
            raise FileExistsError(
                f"Output directory already exists: {model_output}. Use --force to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = params.get(
        "preprocessing",
        {
            "lowercase": True,
            "remove_html_tags": True,
            "normalize_whitespace": True,
        },
    )

    # Use a temporary directory for staging artifacts
    # MLflow will copy these into the model directory with relative paths
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config file in temp directory
        config = {
            "threshold": args.threshold,
            "preprocess_cfg": preprocess_cfg,
            "model_type": params.get("train", {}).get("model_type", "logistic_regression"),
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Copy model and vectorizer to temp directory
        tmp_model_path = tmp_path / "model.pkl"
        tmp_vectorizer_path = tmp_path / "vectorizer.pkl"
        shutil.copy(model_path, tmp_model_path)
        shutil.copy(vectorizer_path, tmp_vectorizer_path)

        print("[save_production_model] Staged artifacts in temp directory")

        # Pass artifacts from temp directory - MLflow will copy them with relative paths
        artifacts = {
            "config": str(config_path),
            "model": str(tmp_model_path),
            "vectorizer": str(tmp_vectorizer_path),
        }

        mlflow.pyfunc.save_model(
            path=str(model_output),
            python_model=SentimentPyfuncModel(),
            artifacts=artifacts,
        )

    print(f"[save_production_model] Saved MLflow pyfunc model -> {model_output}")
    print("[save_production_model] Done.")


if __name__ == "__main__":
    main()
