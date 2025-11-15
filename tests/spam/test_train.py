import json

import torch

from mlops_imdb import config as mlflow_config
from mlops_imdb.spam import train as spam_train


def test_main_trains_and_saves_model(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports" / "figures"
    data_dir.mkdir()
    reports_dir.mkdir(parents=True)

    train_tensor_path = data_dir / "spam_train_features.pt"
    vocab_path = data_dir / "spam_vocab.json"
    model_path = tmp_path / "models" / "spam_model.pt"
    curve_path = reports_dir / "spam_training_loss.png"
    model_path.parent.mkdir(parents=True)

    torch.save(
        {
            "input_ids": torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.long),
            "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
        train_tensor_path,
    )
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"<PAD>": 0, "<UNK>": 1, "word": 2}, f)

    params = {
        "random_seed": 123,
        "spam": {
            "features": {
                "pad_token": "<PAD>",
                "max_len": 3,
                "outputs": {
                    "train_tensor": str(train_tensor_path),
                    "vocab_path": str(vocab_path),
                },
            },
            "train": {
                "embedding_dim": 4,
                "hidden_dim": 4,
                "dropout": 0.0,
                "batch_size": 2,
                "epochs": 2,
                "learning_rate": 0.01,
                "outputs": {
                    "model_path": str(model_path),
                    "training_curve_png": str(curve_path),
                },
            },
        },
        "mlflow": {"experiment_name": "spam-train-test"},
    }

    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    tracking_uri = tracking_dir.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(mlflow_config, "MLFLOW_TRACKING_URI", tracking_uri, raising=False)
    monkeypatch.setattr(mlflow_config, "MLFLOW_EXPERIMENT", "spam-train-test", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    def fake_load_params(path="params.yaml"):
        return params

    monkeypatch.setattr(spam_train, "load_params", fake_load_params)

    spam_train.main()

    assert model_path.exists()
    saved = torch.load(model_path)
    assert "model_state" in saved
    assert curve_path.exists()
