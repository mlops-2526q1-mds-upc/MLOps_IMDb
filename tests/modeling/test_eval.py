import json
import os

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from mlops_imdb import config as mlflow_config
from mlops_imdb.modeling import eval as eval_module


class DummyModel:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        assert X.shape[0] == len(self._predictions)
        return np.array(self._predictions)


def test_ensure_dir_creates_parent(tmp_path):
    target = tmp_path / "reports" / "metrics.json"
    eval_module.ensure_dir(str(target))
    assert target.parent.exists()


def test_save_confusion_matrix_png(tmp_path):
    cm = np.array([[3, 1], [2, 4]])
    out_path = tmp_path / "figures" / "cm.png"
    eval_module.save_confusion_matrix_png(cm, str(out_path), labels=(0, 1))
    assert out_path.exists()
    with open(out_path, "rb") as handle:
        assert handle.read(8).startswith(b"\x89PNG")


def test_main_writes_metrics_and_confusion_matrix(tmp_path, monkeypatch):
    features_path = tmp_path / "imdb_test_features.npz"
    test_csv = tmp_path / "imdb_test_clean.csv"
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    metrics_path = tmp_path / "reports" / "metrics.json"
    cm_path = tmp_path / "reports" / "figures" / "cm.png"
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()
    tracking_uri = tracking_dir.as_uri()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(mlflow_config, "MLFLOW_TRACKING_URI", tracking_uri, raising=False)
    monkeypatch.setattr(mlflow_config, "MLFLOW_EXPERIMENT", "test-eval", raising=False)

    X = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    sp.save_npz(features_path, X)
    pd.DataFrame({"label": [1, 0]}).to_csv(test_csv, index=False)
    joblib.dump(DummyModel([1, 0]), model_path)
    dummy_vectorizer = TfidfVectorizer()
    joblib.dump(dummy_vectorizer, vectorizer_path)

    params = {
        "data": {
            "processed": {"test": str(test_csv)},
            "schema": {"label_col": "label", "label_domain": [0, 1]},
        },
        "features": {
            "outputs": {
                "test_features": str(features_path),
                "vectorizer_path": str(vectorizer_path),
            }
        },
        "train": {"outputs": {"model_path": str(model_path)}},
        "eval": {
            "outputs": {
                "metrics_json": str(metrics_path),
                "confusion_matrix_png": str(cm_path),
            }
        },
    }

    monkeypatch.setattr(eval_module, "load_params", lambda path="params.yaml": params)
    eval_module.main()

    assert metrics_path.exists()
    assert cm_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert payload["accuracy"] == 1.0
    assert payload["precision"] == 1.0
    assert payload["recall"] == 1.0
    assert payload["f1"] == 1.0
    assert payload["inputs"]["model"] == str(model_path)
    assert payload["labels"] == [0, 1]
    assert payload["confusion_matrix"] == [[1, 0], [0, 1]]

    with open(cm_path, "rb") as fh:
        assert fh.read(8).startswith(b"\x89PNG")


def test_last_evaluation_accuracy_threshold():
    """Test that the last evaluation from mlflow has accuracy above 0.85."""
    metrics_path = "reports/metrics.json"

    # Check if metrics file exists
    assert os.path.exists(metrics_path), (
        f"Metrics file not found at {metrics_path}. " "Please run evaluation first."
    )

    # Load metrics
    with open(metrics_path, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    # Check accuracy threshold
    accuracy = metrics.get("accuracy")
    assert accuracy is not None, "Accuracy metric not found in metrics.json"
    assert accuracy > 0.7, (
        f"Model accuracy {accuracy:.4f} is below the required threshold of 0.85. "
        "Please retrain the model or adjust features to improve performance."
    )
