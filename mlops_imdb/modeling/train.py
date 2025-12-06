# src/modeling/train.py
# Purpose: train a Logistic Regression model on sparse TF-IDF features.

from contextlib import nullcontext
import os

from codecarbon import EmissionsTracker
import joblib
import mlflow
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import yaml

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


def main():
    # Load configuration
    params = load_params()

    # Configure MLflow (env or params)
    mlflow_cfg = params.get("mlflow", {})
    configure_mlflow(mlflow_cfg.get("experiment_name"))

    # Start MLflow run and energy tracker
    with mlflow.start_run(run_name="train_model"):
        mlflow.set_tag("stage", "train")
        with create_tracker(params, "train_model") as tracker:
            data_cfg = params["data"]
            schema = data_cfg.get("schema", {"text_col": "text", "label_col": "label"})
            label_col = schema["label_col"]

            train_cfg = params["train"]
            logreg_cfg = train_cfg["logreg"]
            outputs = (
                train_cfg["outputs"]
                if "outputs" in train_cfg
                else {"model_path": "models/model.pkl"}
            )

            # Inputs
            x_train_path = params["features"]["outputs"]["train_features"]
            train_csv_path = data_cfg["processed"]["train"]

            # Output
            model_path = outputs["model_path"]

            # Load data
            X_train = sp.load_npz(x_train_path)
            y_train = pd.read_csv(train_csv_path)[label_col].astype(int).values

            # Define model (solver must support sparse input)
            model = LogisticRegression(
                max_iter=logreg_cfg.get("max_iter", 1000),
                random_state=logreg_cfg.get("random_state", 42),
                solver="liblinear",  # supports sparse; good for binary TF-IDF
                penalty="l2",
            )

            # Train
            model.fit(X_train, y_train)

            # Log params and quick train metric
            mlflow.log_params(
                {
                    "model_type": "logistic_regression",
                    "max_iter": model.max_iter,
                    "random_state": model.random_state,
                    "solver": model.solver,
                    "penalty": model.penalty,
                    "train_rows": int(X_train.shape[0]),
                    "train_cols": int(X_train.shape[1]),
                }
            )
            try:
                train_accuracy = float(model.score(X_train, y_train))
                mlflow.log_metric("train_accuracy", train_accuracy)
            except Exception:
                pass

            # Persist model and log to MLflow
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            print(f"[train] Saved model -> {model_path}")

        emissions = getattr(tracker, "final_emissions", None)
        if emissions is not None:
            mlflow.log_metric("emissions_kg", float(emissions))
            print(f"[emissions] train_model: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
