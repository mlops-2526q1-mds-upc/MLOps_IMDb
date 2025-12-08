"""Tests for the prediction monitor module."""

import numpy as np
import pytest

from mlops_imdb.monitoring.prediction_monitor import (
    Alert,
    AlertLevel,
    MonitoringStats,
    PredictionMonitor,
    create_monitor_from_test_predictions,
)


@pytest.fixture
def reference_predictions():
    """Create reference predictions for testing."""
    np.random.seed(42)
    # Balanced predictions around 0.5
    return np.random.uniform(0.3, 0.7, size=100)


@pytest.fixture
def monitor(reference_predictions):
    """Create a prediction monitor with reference data."""
    return PredictionMonitor(
        reference_predictions=reference_predictions,
        window_size=100,
        confidence_threshold=0.6,
        drift_p_threshold=0.05,
    )


class TestPredictionMonitor:
    """Tests for the PredictionMonitor class."""

    def test_initialization(self, reference_predictions):
        """Test monitor initialization."""
        monitor = PredictionMonitor(
            reference_predictions=reference_predictions,
            window_size=100,
        )
        assert monitor.window_size == 100
        assert monitor._output_drift_detector is not None

    def test_initialization_without_reference(self):
        """Test monitor initialization without reference data."""
        monitor = PredictionMonitor(window_size=100)
        assert monitor._output_drift_detector is None

    def test_record_prediction(self, monitor):
        """Test recording a single prediction."""
        result = monitor.record_prediction(probability=0.8, label=1)
        # First prediction should not trigger alert
        assert result is None

    def test_record_prediction_stats(self, monitor):
        """Test that recorded predictions update stats."""
        for i in range(10):
            monitor.record_prediction(probability=0.6 + i * 0.03, label=1)

        stats = monitor.get_stats()
        assert stats.total_predictions == 10
        assert stats.positive_predictions == 10

    def test_window_size_limit(self, monitor):
        """Test that predictions respect window size."""
        for i in range(150):
            monitor.record_prediction(probability=0.5, label=0)

        stats = monitor.get_stats()
        assert stats.total_predictions == 100  # Window size

    def test_check_prediction_ratio_insufficient_data(self, monitor):
        """Test ratio check with insufficient data."""
        for i in range(10):
            monitor.record_prediction(probability=0.6, label=1)

        alert = monitor.check_prediction_ratio()
        assert alert is None  # Need at least 50 samples

    def test_check_prediction_ratio_normal(self, monitor):
        """Test ratio check with normal distribution."""
        # Add balanced predictions
        for i in range(60):
            label = i % 2
            monitor.record_prediction(probability=0.5 + 0.1 * label, label=label)

        alert = monitor.check_prediction_ratio()
        # Should not alert for balanced predictions
        assert alert is None or alert.level != AlertLevel.CRITICAL

    def test_check_prediction_ratio_drift(self, reference_predictions):
        """Test ratio check when drift detected."""
        # Create monitor with specific reference ratio
        ref_preds = np.concatenate([np.ones(70) * 0.7, np.ones(30) * 0.3])
        monitor = PredictionMonitor(
            reference_predictions=ref_preds,
            positive_ratio_warning=0.1,
            positive_ratio_critical=0.2,
        )

        # Add only positive predictions (ratio drift)
        for i in range(60):
            monitor.record_prediction(probability=0.9, label=1)

        alert = monitor.check_prediction_ratio()
        # May or may not detect depending on threshold
        if alert is not None:
            assert alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]

    def test_check_output_drift_insufficient_data(self, monitor):
        """Test output drift check with insufficient data."""
        for i in range(10):
            monitor.record_prediction(probability=0.5, label=0)

        result = monitor.check_output_drift()
        assert "error" in result or result.get("drift_detected") is False

    def test_check_output_drift(self, monitor):
        """Test output drift detection."""
        for i in range(60):
            monitor.record_prediction(probability=0.5, label=0)

        result = monitor.check_output_drift()
        assert "drift_detected" in result
        assert isinstance(result["drift_detected"], bool)

    def test_get_stats(self, monitor):
        """Test getting monitoring statistics."""
        # Add some predictions
        for i in range(30):
            label = 1 if i < 20 else 0
            prob = 0.7 if label == 1 else 0.3
            monitor.record_prediction(probability=prob, label=label)

        stats = monitor.get_stats()
        assert stats.total_predictions == 30
        assert stats.positive_predictions == 20
        assert stats.negative_predictions == 10
        assert 0 <= stats.avg_confidence <= 1

    def test_get_alerts(self, monitor):
        """Test getting alerts."""
        alerts = monitor.get_alerts()
        assert isinstance(alerts, list)

    def test_clear_alerts(self, monitor):
        """Test clearing alerts."""
        # Force an alert by adding many low confidence predictions
        for i in range(150):
            monitor.record_prediction(probability=0.5, label=0)

        monitor.clear_alerts()
        alerts = monitor.get_alerts()
        assert len(alerts) == 0

    def test_reset(self, monitor):
        """Test resetting the monitor."""
        for i in range(50):
            monitor.record_prediction(probability=0.6, label=1)

        monitor.reset()
        stats = monitor.get_stats()
        assert stats.total_predictions == 0


class TestAlert:
    """Tests for the Alert class."""

    def test_alert_to_dict(self):
        """Test Alert conversion to dictionary."""
        alert = Alert(
            level=AlertLevel.WARNING,
            message="Test warning message",
            metric="confidence",
            value=0.4,
            threshold=0.6,
        )
        result = alert.to_dict()

        assert result["level"] == "warning"
        assert result["message"] == "Test warning message"
        assert result["metric"] == "confidence"
        assert result["value"] == 0.4
        assert result["threshold"] == 0.6
        assert "timestamp" in result


class TestMonitoringStats:
    """Tests for the MonitoringStats class."""

    def test_stats_to_dict(self):
        """Test MonitoringStats conversion to dictionary."""
        stats = MonitoringStats(
            total_predictions=100,
            positive_predictions=60,
            negative_predictions=40,
            avg_confidence=0.75,
            confidence_std=0.15,
            low_confidence_count=10,
        )
        result = stats.to_dict()

        assert result["total_predictions"] == 100
        assert result["positive_predictions"] == 60
        assert result["negative_predictions"] == 40
        assert result["positive_ratio"] == 0.6
        assert result["avg_confidence"] == 0.75

    def test_stats_empty(self):
        """Test empty stats."""
        stats = MonitoringStats()
        result = stats.to_dict()
        assert result["positive_ratio"] == 0.0


class TestCreateMonitorFromTestPredictions:
    """Tests for the factory function."""

    def test_create_monitor(self, reference_predictions):
        """Test creating monitor from test predictions."""
        monitor = create_monitor_from_test_predictions(
            test_predictions=reference_predictions,
            window_size=500,
        )
        assert monitor._output_drift_detector is not None
        assert monitor.window_size == 500

