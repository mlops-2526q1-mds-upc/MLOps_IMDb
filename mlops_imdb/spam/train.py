"""Train the spam classifier using PyTorch + MLflow."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Dict

from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

from mlops_imdb.config import configure_mlflow
from mlops_imdb.spam.mlflow_model import SpamPyfuncModel
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


def save_loss_plot(losses, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, losses, marker="o")
    plt.title("Spam training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    params = load_params()
    spam_cfg = params["spam"]
    features_cfg = spam_cfg["features"]
    train_cfg = spam_cfg["train"]
    feature_out = features_cfg["outputs"]
    train_out = train_cfg["outputs"]

    mlflow_cfg = params.get("mlflow", {})
    configure_mlflow(mlflow_cfg.get("experiment_name"))

    seed = train_cfg.get("random_seed") or params.get("random_seed")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    train_tensor_path = feature_out["train_tensor"]
    vocab_path = feature_out["vocab_path"]
    model_path = train_out["model_path"]
    loss_plot_path = train_out.get("training_curve_png")

    vocab = load_vocab(vocab_path)
    pad_token = features_cfg.get("pad_token", "<PAD>")
    pad_idx = vocab.get(pad_token, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow_tags = {"stage": "spam_train"}
    with mlflow.start_run(run_name="spam_train_model"):
        for key, value in mlflow_tags.items():
            mlflow.set_tag(key, value)
        with create_tracker(params, "spam_train_model") as tracker:
            data_bundle = load_tensor_bundle(train_tensor_path)
            dataset = TensorDataset(data_bundle["input_ids"], data_bundle["labels"])
            train_loader = DataLoader(
                dataset,
                batch_size=train_cfg.get("batch_size", 32),
                shuffle=True,
            )

            model = SpamClassifier(
                vocab_size=len(vocab),
                embed_dim=train_cfg.get("embedding_dim", 64),
                hidden_dim=train_cfg.get("hidden_dim", 64),
                dropout=train_cfg.get("dropout", 0.0),
                padding_idx=pad_idx,
            )

            try:
                model = model.to(device)
            except RuntimeError as exc:
                if device.type == "cuda":
                    print(f"[spam_train] CUDA unavailable ({exc}); falling back to CPU.")
                    device = torch.device("cpu")
                    model = model.to(device)
                else:
                    raise

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 0.001))

            epochs = train_cfg.get("epochs", 5)
            loss_history = []
            best_accuracy = 0.0
            best_epoch = 0
            best_model_state = None

            mlflow.log_params(
                {
                    "model_type": "spam_lstm",
                    "embedding_dim": train_cfg.get("embedding_dim", 64),
                    "hidden_dim": train_cfg.get("hidden_dim", 64),
                    "dropout": train_cfg.get("dropout", 0.0),
                    "learning_rate": train_cfg.get("learning_rate", 0.001),
                    "epochs": epochs,
                    "batch_size": train_cfg.get("batch_size", 32),
                    "vocab_size": len(vocab),
                    "max_len": features_cfg.get("max_len", 50),
                }
            )

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                total = 0
                correct = 0

                for batch_inputs, batch_labels in train_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()
                    preds = model(batch_inputs)
                    loss = criterion(preds, batch_labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * batch_inputs.size(0)
                    total += batch_inputs.size(0)
                    correct += (preds.detach() > 0.5).float().eq(batch_labels).sum().item()

                avg_loss = total_loss / total
                accuracy = correct / total if total else 0.0
                loss_history.append(avg_loss)

                # Save best model checkpoint
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"[spam_train] Epoch {epoch} | Loss {avg_loss:.4f} | Acc {accuracy:.4f} [NEW BEST]")
                else:
                    print(f"[spam_train] Epoch {epoch} | Loss {avg_loss:.4f} | Acc {accuracy:.4f}")

                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("train_accuracy", accuracy, step=epoch)

            # Restore best model weights
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"[spam_train] Restored best model from epoch {best_epoch} (Acc {best_accuracy:.4f})")
                mlflow.log_metric("best_epoch", best_epoch)
                mlflow.log_metric("best_accuracy", best_accuracy)

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_path": vocab_path,
                    "pad_idx": pad_idx,
                },
                model_path,
            )
            mlflow.log_artifact(model_path, artifact_path="spam_model_raw")

            infer_config = {
                "pad_token": pad_token,
                "unk_token": features_cfg.get("unk_token", "<UNK>"),
                "max_len": features_cfg.get("max_len", 50),
                "pad_idx": pad_idx,
                "embedding_dim": train_cfg.get("embedding_dim", 64),
                "hidden_dim": train_cfg.get("hidden_dim", 64),
                "dropout": train_cfg.get("dropout", 0.0),
                "threshold": train_cfg.get("threshold", 0.5),
                "preprocess_cfg": spam_cfg.get("preprocessing", {}),
            }

            signature = ModelSignature(
                inputs=Schema([ColSpec(DataType.string, "text")]),
                outputs=Schema([ColSpec(DataType.double, "prediction")]),
            )

            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                state_path = tmpdir_path / "state_dict.pt"
                vocab_copy = tmpdir_path / "vocab.json"
                config_path = tmpdir_path / "config.json"
                pyfunc_dir = tmpdir_path / "spam_model"

                torch.save(model.state_dict(), state_path)
                shutil.copy(vocab_path, vocab_copy)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(infer_config, f, indent=2)

                mlflow.pyfunc.save_model(
                    path=str(pyfunc_dir),
                    python_model=SpamPyfuncModel(),
                    artifacts={
                        "state_dict": str(state_path),
                        "vocab": str(vocab_copy),
                        "config": str(config_path),
                    },
                    signature=signature,
                    pip_requirements=[
                        "mlflow",
                        "torch>=2.9.1",
                        "pandas",
                        "pyyaml",
                        "numpy",
                    ],
                )
                mlflow.log_artifacts(str(pyfunc_dir), artifact_path="spam_model")

            if loss_plot_path and loss_history:
                save_loss_plot(loss_history, loss_plot_path)
                try:
                    mlflow.log_artifact(loss_plot_path, artifact_path="spam_train")
                except Exception:
                    pass

        emissions = getattr(tracker, "final_emissions", None)
        if emissions is not None:
            mlflow.log_metric("emissions_kg", float(emissions))
            print(f"[emissions] spam_train_model: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
