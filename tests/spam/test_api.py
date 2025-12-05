"""Tests for the spam prediction API endpoints."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_imdb.spam import api as spam_api


class MockModel:
    """Mock pyfunc model for testing."""

    def predict(self, df: pd.DataFrame):
        """Return a fixed probability for testing."""
        text = df["text"].iloc[0] if "text" in df.columns else df.iloc[0, 0]
        if "spam" in text.lower() or "buy now" in text.lower():
            return [0.95]
        return [0.15]


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a mocked model."""
    monkeypatch.setattr(spam_api, "model", MockModel())
    return TestClient(spam_api.app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_uri" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_spam_text(self, client):
        """Predict endpoint should classify spam-like text as spam."""
        response = client.post(
            "/predict",
            json={"text": "BUY NOW! Free money! Click here!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "label" in data
        assert data["label"] == 1
        assert data["probability"] >= 0.5

    def test_predict_ham_text(self, client):
        """Predict endpoint should classify normal text as not spam."""
        response = client.post(
            "/predict",
            json={"text": "Hello, how are you doing today?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == 0
        assert data["probability"] < 0.5

    def test_predict_returns_probability_in_range(self, client):
        """Probability should be between 0 and 1."""
        response = client.post(
            "/predict",
            json={"text": "Test message"},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_missing_text_field(self, client):
        """Predict endpoint should return 422 for missing text field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json(self, client):
        """Predict endpoint should return 422 for invalid JSON."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_empty_text(self, client):
        """Predict endpoint should handle empty text."""
        response = client.post(
            "/predict",
            json={"text": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "label" in data

