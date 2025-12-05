"""Tests for the sentiment prediction API endpoints."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_imdb.modeling import api as sentiment_api


class MockModel:
    """Mock pyfunc model for testing."""

    def predict(self, df: pd.DataFrame):
        """Return a fixed probability based on text content for testing."""
        text = df["text"].iloc[0] if "text" in df.columns else df.iloc[0, 0]
        text_lower = text.lower()
        if any(word in text_lower for word in ["great", "amazing", "excellent", "love"]):
            return [0.92]
        if any(word in text_lower for word in ["terrible", "awful", "hate", "boring"]):
            return [0.12]
        return [0.50]


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a mocked model."""
    monkeypatch.setattr(sentiment_api, "model", MockModel())
    return TestClient(sentiment_api.app)


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

    def test_predict_positive_sentiment(self, client):
        """Predict endpoint should classify positive text correctly."""
        response = client.post(
            "/predict",
            json={"text": "This movie was amazing! I absolutely loved it."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "label" in data
        assert "sentiment" in data
        assert data["label"] == 1
        assert data["sentiment"] == "positive"
        assert data["probability"] >= 0.5

    def test_predict_negative_sentiment(self, client):
        """Predict endpoint should classify negative text correctly."""
        response = client.post(
            "/predict",
            json={"text": "This movie was terrible and boring. I hate it."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == 0
        assert data["sentiment"] == "negative"
        assert data["probability"] < 0.5

    def test_predict_returns_probability_in_range(self, client):
        """Probability should be between 0 and 1."""
        response = client.post(
            "/predict",
            json={"text": "This is a neutral test message."},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_returns_valid_label(self, client):
        """Label should be either 0 or 1."""
        response = client.post(
            "/predict",
            json={"text": "Some random text here."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in [0, 1]

    def test_predict_returns_valid_sentiment(self, client):
        """Sentiment should be either positive or negative."""
        response = client.post(
            "/predict",
            json={"text": "Some random text here."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative"]

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
        assert "sentiment" in data

    def test_predict_long_text(self, client):
        """Predict endpoint should handle long text input."""
        long_text = "This is a great movie. " * 100
        response = client.post(
            "/predict",
            json={"text": long_text},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == 1
        assert data["sentiment"] == "positive"

    def test_predict_special_characters(self, client):
        """Predict endpoint should handle text with special characters."""
        response = client.post(
            "/predict",
            json={"text": "Great movie!!! <br> & special chars @#$%"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data

