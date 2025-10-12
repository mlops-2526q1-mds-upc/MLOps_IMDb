# src/modeling/train.py
# Purpose: train a Logistic Regression model on sparse TF-IDF features.

from codecarbon import EmissionsTracker
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    with EmissionsTracker(project_name="train_model") as tracker:
        # Load configuration
        params = load_params()
        data_cfg = params["data"]
        schema = data_cfg.get("schema", {"text_col": "text", "label_col": "label"})
        label_col = schema["label_col"]

        train_cfg = params["train"]
        logreg_cfg = train_cfg["logreg"]
        outputs = (
            train_cfg["outputs"] if "outputs" in train_cfg else {"model_path": "models/model.pkl"}
        )

        # Inputs
        X_train_path = params["features"]["outputs"]["train_features"]
        train_csv_path = data_cfg["processed"]["train"]

        # Output
        model_path = outputs["model_path"]

        # Load data
        X_train = sp.load_npz(X_train_path)
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

        # Persist model
        joblib.dump(model, model_path)
        print(f"[train] Saved model -> {model_path}")
    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] train_model: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
