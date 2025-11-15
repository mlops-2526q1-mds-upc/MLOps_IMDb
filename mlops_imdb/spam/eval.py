"""Evaluate the spam classifier and log metrics."""

from contextlib import nullcontext
import json
import os
from typing import Dict

from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from mlops_imdb.config import configure_mlflow
from mlops_imdb.spam.model import SpamClassifier


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_tracker(params: dict, project_name: str):
    energy_cfg = params.get("energy", {}).get("codecarbon", {})
    if not energy_cfg.get("enabled"):
        return nullcontext()
    output_path = energy_cfg.get("output", "reports/codecarbon_emissions.csv")
    output_dir, output_file = os.path.split(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."
    return EmissionsTracker(
        project_name=project_name, output_dir=output_dir, output_file=output_file
    )


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tensor_bundle(path: str) -> Dict[str, torch.Tensor]:
    bundle = torch.load(path, map_location="cpu")
    return {
        "input_ids": bundle["input_ids"],
        "labels": bundle["labels"],
    }


def load_model_state(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def save_confusion_matrix(cm, out_path: str, labels):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Spam Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    params = load_params()
    spam_cfg = params["spam"]
    features_cfg = spam_cfg["features"]
    eval_cfg = spam_cfg["eval"]
    train_cfg = spam_cfg["train"]
    data_cfg = spam_cfg["data"]
    schema = data_cfg.get(
        "schema",
        {"text_col": "text", "label_col": "label", "label_domain": [0, 1]},
    )
    label_domain = schema.get("label_domain", [0, 1])

    feature_out = features_cfg["outputs"]
    test_tensor_path = feature_out["test_tensor"]
    vocab_path = feature_out["vocab_path"]
    model_path = train_cfg["outputs"]["model_path"]

    metrics_path = eval_cfg["outputs"]["metrics_json"]
    cm_path = eval_cfg["outputs"]["confusion_matrix_png"]

    mlflow_cfg = params.get("mlflow", {})
    configure_mlflow(mlflow_cfg.get("experiment_name"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow_tags = {"stage": "spam_eval"}
    with mlflow.start_run(run_name="spam_evaluate_model"):
        for key, value in mlflow_tags.items():
            mlflow.set_tag(key, value)
        with create_tracker(params, "spam_evaluate_model") as tracker:
            vocab = load_vocab(vocab_path)
            pad_idx = vocab.get(features_cfg.get("pad_token", "<PAD>"), 0)

            state_bundle = load_model_state(model_path)
            model = SpamClassifier(
                vocab_size=len(vocab),
                embed_dim=train_cfg.get("embedding_dim", 64),
                hidden_dim=train_cfg.get("hidden_dim", 64),
                dropout=train_cfg.get("dropout", 0.0),
                padding_idx=pad_idx,
            )
            model.load_state_dict(state_bundle["model_state"])
            model.to(device)
            model.eval()

            data_bundle = load_tensor_bundle(test_tensor_path)
            dataset = TensorDataset(data_bundle["input_ids"], data_bundle["labels"])
            loader = DataLoader(dataset, batch_size=eval_cfg.get("batch_size", 64))

            preds_list = []
            labels_list = []
            with torch.no_grad():
                for batch_inputs, batch_labels in loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)
                    preds = model(batch_inputs)
                    preds_list.append(preds.cpu())
                    labels_list.append(batch_labels.cpu())

            y_pred = torch.cat(preds_list).numpy()
            y_true = torch.cat(labels_list).numpy()
            y_pred_cls = (y_pred > 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred_cls)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_cls, average="binary", zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred_cls, labels=label_domain)

            mlflow.log_metrics(
                {
                    "accuracy": float(acc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            payload = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "labels": label_domain,
                "confusion_matrix": [[int(v) for v in row] for row in cm.tolist()],
                "inputs": {
                    "test_tensor": test_tensor_path,
                    "model_path": model_path,
                    "vocab_path": vocab_path,
                },
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            try:
                mlflow.log_artifact(metrics_path, artifact_path="spam_eval")
            except Exception:
                pass

            save_confusion_matrix(cm, cm_path, labels=label_domain)
            try:
                mlflow.log_artifact(cm_path, artifact_path="spam_eval")
            except Exception:
                pass

        emissions = getattr(tracker, "final_emissions", None)
        if emissions is not None:
            mlflow.log_metric("emissions_kg", float(emissions))
            print(f"[emissions] spam_evaluate_model: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
