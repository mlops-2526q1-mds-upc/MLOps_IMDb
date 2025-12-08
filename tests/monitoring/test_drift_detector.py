"""Tests for the drift detector module."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from mlops_imdb.monitoring.drift_detector import (
    DriftDetector,
    DriftReport,
    DriftStatus,
    create_detector_from_training_data,
)


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "This is a great movie with excellent acting.",
        "Terrible film, waste of time and money.",
        "The plot was interesting but the ending was disappointing.",
        "Amazing performance by the lead actor.",
        "Boring and predictable storyline throughout.",
        "One of the best films I have ever seen.",
        "The cinematography was beautiful and stunning.",
        "Horrible script and terrible direction.",
        "A masterpiece of modern cinema.",
        "Would not recommend this to anyone.",
    ] * 10  # Repeat to get enough samples


@pytest.fixture
def fitted_vectorizer(sample_texts):
    """Create a fitted TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=100)
    vectorizer.fit(sample_texts)
    return vectorizer


@pytest.fixture
def reference_features(sample_texts, fitted_vectorizer):
    """Create reference features from sample texts."""
    features = fitted_vectorizer.transform(sample_texts)
    return features.toarray()


class TestDriftDetector:
    """Tests for the DriftDetector class."""

    def test_initialization_with_reference_data(self, reference_features, fitted_vectorizer):
        """Test detector initialization with reference data."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            p_threshold=0.05,
        )
        assert detector._initialized is True
        assert detector.vectorizer is not None

    def test_initialization_without_reference(self, fitted_vectorizer):
        """Test detector initialization without reference data."""
        detector = DriftDetector(
            vectorizer=fitted_vectorizer,
            p_threshold=0.05,
        )
        assert detector._initialized is False

    def test_set_reference_data(self, fitted_vectorizer, reference_features):
        """Test setting reference data after initialization."""
        detector = DriftDetector(vectorizer=fitted_vectorizer)
        assert detector._initialized is False

        detector.set_reference_data(reference_features=reference_features)
        assert detector._initialized is True

    def test_add_sample(self, reference_features, fitted_vectorizer):
        """Test adding samples to buffer."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            window_size=100,
        )
        detector.add_sample("Test sample text")
        assert detector.get_buffer_size() == 1

    def test_add_samples_batch(self, reference_features, fitted_vectorizer):
        """Test adding batch samples to buffer."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            window_size=100,
        )
        texts = ["Sample one", "Sample two", "Sample three"]
        detector.add_samples_batch(texts)
        assert detector.get_buffer_size() == 3

    def test_buffer_window_limit(self, reference_features, fitted_vectorizer):
        """Test that buffer respects window size."""
        window_size = 5
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            window_size=window_size,
        )

        # Add more samples than window size
        for i in range(10):
            detector.add_sample(f"Sample text number {i}")

        assert detector.get_buffer_size() == window_size

    def test_detect_drift_insufficient_data(self, reference_features, fitted_vectorizer):
        """Test drift detection with insufficient data."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            min_samples=50,
        )
        detector.add_sample("Single sample")

        report = detector.detect_drift(use_buffer=True)
        assert report.status == DriftStatus.INSUFFICIENT_DATA

    def test_detect_drift_no_drift(self, reference_features, fitted_vectorizer, sample_texts):
        """Test drift detection with similar data (no drift expected)."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            p_threshold=0.05,
            min_samples=10,
        )

        # Use same distribution as reference
        report = detector.detect_drift(texts=sample_texts)
        assert report.status in [DriftStatus.NO_DRIFT, DriftStatus.DRIFT_DETECTED]
        assert report.p_value is not None

    def test_detect_drift_with_drift(self, reference_features, fitted_vectorizer):
        """Test drift detection with different data distribution."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
            p_threshold=0.05,
            min_samples=10,
        )

        # Create very different data
        different_texts = [
            "Technical documentation about software engineering.",
            "Medical research findings on cardiovascular health.",
            "Financial analysis of quarterly earnings reports.",
            "Legal contract terms and conditions specified herein.",
        ] * 25

        report = detector.detect_drift(texts=different_texts)
        assert report.status in [DriftStatus.NO_DRIFT, DriftStatus.DRIFT_DETECTED]
        assert report.sample_size == 100

    def test_detect_drift_not_initialized(self, fitted_vectorizer):
        """Test drift detection when not initialized."""
        detector = DriftDetector(vectorizer=fitted_vectorizer)
        report = detector.detect_drift(texts=["Test"])
        assert report.status == DriftStatus.ERROR

    def test_clear_buffer(self, reference_features, fitted_vectorizer):
        """Test clearing the sample buffer."""
        detector = DriftDetector(
            reference_data=reference_features,
            vectorizer=fitted_vectorizer,
        )
        detector.add_sample("Test sample")
        assert detector.get_buffer_size() == 1

        detector.clear_buffer()
        assert detector.get_buffer_size() == 0


class TestDriftReport:
    """Tests for the DriftReport class."""

    def test_drift_report_to_dict(self):
        """Test DriftReport conversion to dictionary."""
        report = DriftReport(
            status=DriftStatus.DRIFT_DETECTED,
            p_value=0.01,
            threshold=0.05,
            drift_score=0.8,
            sample_size=100,
        )
        result = report.to_dict()

        assert result["status"] == "drift_detected"
        assert result["p_value"] == 0.01
        assert result["threshold"] == 0.05
        assert result["drift_detected"] is True
        assert result["sample_size"] == 100

    def test_drift_report_no_drift(self):
        """Test DriftReport for no drift case."""
        report = DriftReport(
            status=DriftStatus.NO_DRIFT,
            p_value=0.15,
            threshold=0.05,
        )
        result = report.to_dict()

        assert result["drift_detected"] is False
        assert result["status"] == "no_drift"


class TestCreateDetectorFromTrainingData:
    """Tests for the factory function."""

    def test_create_detector(self, sample_texts, fitted_vectorizer):
        """Test creating detector from training data."""
        detector = create_detector_from_training_data(
            train_texts=sample_texts,
            vectorizer=fitted_vectorizer,
            p_threshold=0.05,
        )
        assert detector._initialized is True
        assert detector.vectorizer is not None

    def test_create_detector_with_sampling(self, sample_texts, fitted_vectorizer):
        """Test creating detector with sample size limit."""
        detector = create_detector_from_training_data(
            train_texts=sample_texts,
            vectorizer=fitted_vectorizer,
            sample_size=10,
        )
        assert detector._initialized is True

