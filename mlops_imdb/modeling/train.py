from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import joblib
from loguru import logger
from tqdm import tqdm
import typer

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

from mlops_imdb.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    configure_mlflow,
    ENV_NAME,
    DAGSHUB_REPO,
)

app = typer.Typer()


@dataclass
class TrainConfig:
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    C: float = 2.0
    max_iter: int = 1000
    test_size: float = 0.2
    random_state: int = 42


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    run_name: str = "train",
):
    """
    Train a text classifier (TF-IDF + LogisticRegression),
    log to MLflow (DagsHub), and save a local model.pkl.
    """
    # Configure MLflow (no runs started here)
    configure_mlflow(experiment_name="mlops-imdb")

    cfg = TrainConfig()
    logger.info("Loading data (texts & labels)...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]

    if "text" not in X.columns:
        raise ValueError(f"'text' column not found in {features_path}")
    X["text"] = X["text"].fillna("")

    logger.info("Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X["text"],
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.max_features,
                    ngram_range=cfg.ngram_range,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg.C,
                    max_iter=cfg.max_iter,
                    random_state=cfg.random_state,
                    n_jobs=None,
                ),
            ),
        ]
    )

    # Exactly ONE run, opened here
    with mlflow.start_run(run_name=run_name):
        print("[MLflow] run_id:", mlflow.active_run().info.run_id)

        # set helpful tags INSIDE the run
        mlflow.set_tags(
            {
                "env": ENV_NAME,
                "repo": DAGSHUB_REPO or "",
                "component": "training",
            }
        )

        # Log hyperparameters
        mlflow.log_params(
            {
                "max_features": cfg.max_features,
                "ngram_range": str(cfg.ngram_range),
                "C": cfg.C,
                "max_iter": cfg.max_iter,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
            }
        )

        logger.info("Training model (TF-IDF + LogisticRegression)...")
        for _ in tqdm(range(1), total=1):
            pipeline.fit(X_train, y_train)

        logger.info("Evaluating...")
        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary"
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", p)
        mlflow.log_metric("recall", r)
        mlflow.log_metric("f1", f1)

        logger.success(
            f"Validation metrics — "
            f"accuracy: {acc:.4f}, precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}"
        )

        # Log model to MLflow with signature and example
        sig = infer_signature(X_val.to_frame(name="text"), y_pred)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=sig,
            input_example=X_val.iloc[:3].to_frame(name="text"),
        )

        # Log input CSVs for traceability
        mlflow.log_artifact(str(features_path))
        mlflow.log_artifact(str(labels_path))

        # Save local copy
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        logger.success(f"Saved local model to {model_path}")

    logger.success("Training complete.")


if __name__ == "__main__":
    app()
