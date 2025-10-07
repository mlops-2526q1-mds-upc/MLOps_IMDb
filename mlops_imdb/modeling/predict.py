from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
from loguru import logger
from tqdm import tqdm
import typer

import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from mlops_imdb.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    configure_mlflow,
    ENV_NAME,
    DAGSHUB_REPO,
)

app = typer.Typer()


def _infer_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute basic metrics. Uses 'binary' if 2 classes, else 'weighted'."""
    avg = "binary" if y_true.nunique() == 2 else "weighted"
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    labels_path: Optional[Path] = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    run_name: str = "predict",
):
    """
    Load the local sklearn Pipeline (TF-IDF + LR), run inference on test texts,
    save predictions, and log artifacts/metrics to MLflow (DagsHub).
    """
    # Configure MLflow (no runs started here)
    configure_mlflow(experiment_name="mlops-imdb")

    logger.info("Loading test features and model...")
    X = pd.read_csv(features_path)  # expects a 'text' column
    if "text" not in X.columns:
        raise ValueError(f"'text' column not found in {features_path}")
    X["text"] = X["text"].fillna("")

    model = joblib.load(model_path)

    # Exactly ONE run, opened here
    with mlflow.start_run(run_name=run_name):
        print("[MLflow] run_id:", mlflow.active_run().info.run_id)

        # set helpful tags INSIDE the run
        mlflow.set_tags(
            {
                "env": ENV_NAME,
                "repo": DAGSHUB_REPO or "",
                "component": "inference",
            }
        )

        # context
        mlflow.log_params(
            {
                "features_rows": int(X.shape[0]),
                "features_cols": int(X.shape[1]),
                "used_model_path": str(model_path),
            }
        )

        logger.info("Running inference...")
        for _ in tqdm(range(1), total=1):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X["text"])
                if proba.shape[1] == 2:
                    preds_out = pd.DataFrame(
                        {
                            "proba_neg": proba[:, 0],
                            "proba_pos": proba[:, 1],
                            "prediction": (proba[:, 1] >= 0.5).astype(int),
                        }
                    )
                else:
                    pred_labels = proba.argmax(axis=1)
                    preds_out = pd.DataFrame(proba)
                    preds_out.columns = [f"proba_{i}" for i in range(proba.shape[1])]
                    preds_out["prediction"] = pred_labels
            else:
                pred = model.predict(X["text"])
                preds_out = pd.DataFrame({"prediction": pred})

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        preds_out.to_csv(predictions_path, index=False)
        logger.success(f"Saved predictions to {predictions_path}")
        mlflow.log_artifact(str(predictions_path))

        # Optional metrics if labels available
        if labels_path and labels_path.exists():
            logger.info(f"Labels found at {labels_path}, computing metrics...")
            y_true = pd.read_csv(labels_path).iloc[:, 0]
            y_pred = preds_out["prediction"]
            metrics = _infer_metrics(y_true, y_pred)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            logger.success(
                "Metrics — " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )

            # Save a text classification report
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = REPORTS_DIR / "classification_report.txt"
            report_path.write_text(
                classification_report(y_true, y_pred, digits=4),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path))
        else:
            logger.warning(
                "Labels file not found — metrics will not be computed. "
                f"Expected at: {labels_path}"
            )

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
