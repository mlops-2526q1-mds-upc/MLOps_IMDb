# src/modeling/eval.py
# Purpose: evaluate the trained model on test features and write metrics JSON (+ confusion matrix PNG).

from contextlib import nullcontext
import json
import os

from codecarbon import EmissionsTracker
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import yaml
import mlflow

from mlops_imdb.config import configure_mlflow


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_tracker(params: dict, project_name: str):
    energy_cfg = params.get("energy", {}).get("codecarbon", {})
    if not energy_cfg.get("enabled"):
        return nullcontext()
    output_path = energy_cfg.get("output", "emissions.csv")
    output_dir, output_file = os.path.split(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."
    return EmissionsTracker(
        project_name=project_name, output_dir=output_dir, output_file=output_file
    )


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_confusion_matrix_png(cm, out_path: str, labels=(0, 1)) -> None:
    ensure_dir(out_path)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    params = load_params()

    # Configure MLflow (env or params)
    mlflow_cfg = params.get("mlflow", {})
    configure_mlflow(mlflow_cfg.get("experiment_name"))

    # Start MLflow run and energy tracker
    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.set_tag("stage", "eval")
        with create_tracker(params, "evaluate_model") as tracker:
            # Paths and schema
            data_cfg = params["data"]
            schema = data_cfg.get(
                "schema", {"text_col": "text", "label_col": "label", "label_domain": [0, 1]}
            )
            label_col = schema["label_col"]
            label_domain = schema.get("label_domain", [0, 1])

        features_out = params["features"]["outputs"]
        eval_out = params.get("eval", {}).get("outputs", {})
        metrics_json = eval_out.get("metrics_json", "reports/metrics.json")
        cm_png = eval_out.get(
            "confusion_matrix_png", "reports/figures/baseline_confusion_matrix.png"
        )

        # Inputs
        X_test_path = features_out["test_features"]
        test_csv_path = data_cfg["processed"]["test"]
        model_path = params["train"]["outputs"]["model_path"]

        # Load data
        X_test = sp.load_npz(X_test_path)
        y_test = pd.read_csv(test_csv_path)[label_col].astype(int).values

        # Load model
        model = joblib.load(model_path)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        mlflow.log_params(
            {
                "test_rows": int(X_test.shape[0]),
                "test_cols": int(X_test.shape[1]),
            }
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=label_domain)
        save_confusion_matrix_png(cm, cm_png, labels=tuple(label_domain))

        # Persist metrics JSON
        ensure_dir(metrics_json)
        payload = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "labels": label_domain,
            "confusion_matrix": [[int(v) for v in row] for row in cm.tolist()],
            "inputs": {
                "X_test": X_test_path,
                "y_test_csv": test_csv_path,
                "model": model_path,
            },
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval] Saved metrics -> {metrics_json}")
        print(f"[eval] Saved confusion matrix -> {cm_png}")
        # Log artifacts to MLflow
        try:
            mlflow.log_artifact(metrics_json, artifact_path="eval")
        except Exception:
            pass
        try:
            mlflow.log_artifact(cm_png, artifact_path="eval")
        except Exception:
            pass

        # After tracker closes, log emissions if available
        emissions = getattr(tracker, "final_emissions", None)
    # tracker context has exited here
    emissions = locals().get("emissions", None)
    if emissions is not None:
        mlflow.log_metric("emissions_kg", float(emissions))
        print(f"[emissions] evaluate_model: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
