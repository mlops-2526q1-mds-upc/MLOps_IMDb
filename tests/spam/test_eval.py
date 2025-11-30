import json

import torch
import torch.nn as nn

from mlops_imdb import config as mlflow_config
from mlops_imdb.spam import eval as spam_eval


def test_main_evaluates_and_writes_metrics(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports" / "figures"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    reports_dir.mkdir(parents=True)
    models_dir.mkdir()

    test_tensor_path = data_dir / "spam_test_features.pt"
    vocab_path = data_dir / "spam_vocab.json"
    model_path = models_dir / "spam_model.pt"
    metrics_path = tmp_path / "reports" / "spam_metrics.json"
    cm_path = reports_dir / "spam_confusion_matrix.png"

    torch.save(
        {
            "input_ids": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
            "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
        test_tensor_path,
    )
    vocab = {"<PAD>": 0, "<UNK>": 1}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    class DummyClassifier(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            batch = x.shape[0]
            return torch.sigmoid(self.bias.expand(batch))

    model = DummyClassifier()
    model.bias.data.fill_(5.0)
    torch.save(
        {"model_state": model.state_dict(), "vocab_path": str(vocab_path), "pad_idx": 0},
        model_path,
    )

    params = {
        "spam": {
            "data": {"schema": {"label_domain": [0, 1]}},
            "features": {
                "pad_token": "<PAD>",
                "outputs": {
                    "test_tensor": str(test_tensor_path),
                    "vocab_path": str(vocab_path),
                },
            },
            "train": {
                "embedding_dim": 4,
                "hidden_dim": 4,
                "dropout": 0.0,
                "outputs": {"model_path": str(model_path)},
            },
            "eval": {
                "batch_size": 1,
                "outputs": {
                    "metrics_json": str(metrics_path),
                    "confusion_matrix_png": str(cm_path),
                },
            },
        },
        "mlflow": {"experiment_name": "spam-eval-test"},
    }

    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    tracking_uri = tracking_dir.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(mlflow_config, "MLFLOW_TRACKING_URI", tracking_uri, raising=False)
    monkeypatch.setattr(mlflow_config, "MLFLOW_EXPERIMENT", "spam-eval-test", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)

    monkeypatch.setattr(spam_eval, "load_params", lambda path="params.yaml": params)
    monkeypatch.setattr(spam_eval, "SpamClassifier", DummyClassifier)

    spam_eval.main()

    assert metrics_path.exists()
    assert cm_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in payload
    assert payload["labels"] == [0, 1]
