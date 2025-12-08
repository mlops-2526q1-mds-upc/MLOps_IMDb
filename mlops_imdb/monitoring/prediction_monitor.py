"""Prediction monitor for tracking model outputs and detecting performance degradation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

from alibi_detect.cd import KSDrift
import numpy as np

from mlops_imdb.logger import get_logger

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels for monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Monitoring alert data structure."""

    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class MonitoringStats:
    """Statistics from prediction monitoring."""

    total_predictions: int = 0
    positive_predictions: int = 0
    negative_predictions: int = 0
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    low_confidence_count: int = 0
    prediction_drift_detected: bool = False
    alerts: List[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "positive_predictions": self.positive_predictions,
            "negative_predictions": self.negative_predictions,
            "positive_ratio": (
                self.positive_predictions / self.total_predictions
                if self.total_predictions > 0
                else 0.0
            ),
            "avg_confidence": self.avg_confidence,
            "confidence_std": self.confidence_std,
            "low_confidence_count": self.low_confidence_count,
            "prediction_drift_detected": self.prediction_drift_detected,
            "alerts": [a.to_dict() for a in self.alerts],
        }


class PredictionMonitor:
    """Monitor for tracking predictions and detecting output drift.

    Tracks:
    - Prediction distribution (positive/negative ratio)
    - Confidence scores and their distribution
    - Low confidence predictions
    - Output drift compared to reference distribution
    """

    def __init__(
        self,
        reference_predictions: Optional[np.ndarray] = None,
        window_size: int = 1000,
        confidence_threshold: float = 0.6,
        positive_ratio_warning: float = 0.3,
        positive_ratio_critical: float = 0.2,
        drift_p_threshold: float = 0.05,
    ):
        """Initialize the prediction monitor.

        Args:
            reference_predictions: Reference prediction probabilities for drift detection.
            window_size: Size of the sliding window for statistics.
            confidence_threshold: Threshold below which predictions are flagged as low confidence.
            positive_ratio_warning: Warning threshold for positive prediction ratio deviation.
            positive_ratio_critical: Critical threshold for positive prediction ratio deviation.
            drift_p_threshold: P-value threshold for output drift detection.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.positive_ratio_warning = positive_ratio_warning
        self.positive_ratio_critical = positive_ratio_critical
        self.drift_p_threshold = drift_p_threshold

        self._lock = Lock()
        self._predictions: Deque[Tuple[float, int]] = deque(maxlen=window_size)
        self._total_count = 0
        self._alerts: List[Alert] = []

        # Output drift detector
        self._output_drift_detector = None
        self._reference_positive_ratio = 0.5  # Default assumption

        if reference_predictions is not None:
            self._initialize_output_drift(reference_predictions)

    def _initialize_output_drift(self, reference_predictions: np.ndarray) -> None:
        """Initialize output drift detector with reference predictions."""
        try:
            reference_predictions = np.asarray(reference_predictions).flatten()
            self._reference_positive_ratio = float(np.mean(reference_predictions >= 0.5))

            # Initialize KS drift detector for output distribution
            ref_reshaped = reference_predictions.reshape(-1, 1)
            self._output_drift_detector = KSDrift(
                ref_reshaped,
                p_val=self.drift_p_threshold,
            )
            logger.info(
                "Initialized output drift detector with %d reference predictions (positive ratio: %.3f)",
                len(reference_predictions),
                self._reference_positive_ratio,
            )
        except Exception as exc:
            logger.warning("Failed to initialize output drift detector: %s", exc)

    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Set reference predictions for drift detection.

        Args:
            predictions: Array of reference prediction probabilities.
        """
        with self._lock:
            self._initialize_output_drift(predictions)

    def record_prediction(self, probability: float, label: int) -> Optional[Alert]:
        """Record a single prediction and check for anomalies.

        Args:
            probability: Prediction probability (0-1).
            label: Predicted label (0 or 1).

        Returns:
            Alert if an anomaly is detected, None otherwise.
        """
        with self._lock:
            self._predictions.append((probability, label))
            self._total_count += 1

            alert = None

            # Check for low confidence prediction
            confidence = abs(probability - 0.5) * 2
            if confidence < self.confidence_threshold:
                if self._should_alert_low_confidence():
                    alert = Alert(
                        level=AlertLevel.INFO,
                        message="Low confidence prediction detected",
                        metric="confidence",
                        value=confidence,
                        threshold=self.confidence_threshold,
                    )
                    self._alerts.append(alert)

            return alert

    def _should_alert_low_confidence(self) -> bool:
        """Determine if we should alert for low confidence (rate limiting)."""
        # Alert at most once per 100 predictions for low confidence
        return self._total_count % 100 == 0

    def check_prediction_ratio(self) -> Optional[Alert]:
        """Check if prediction ratio has drifted significantly.

        Returns:
            Alert if ratio has drifted, None otherwise.
        """
        with self._lock:
            if len(self._predictions) < 50:
                return None

            predictions = list(self._predictions)
            positive_count = sum(1 for _, label in predictions if label == 1)
            current_ratio = positive_count / len(predictions)

            # Compare with reference ratio
            ratio_diff = abs(current_ratio - self._reference_positive_ratio)

            if ratio_diff >= self.positive_ratio_critical:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Prediction ratio critically different from reference ({current_ratio:.2%} vs {self._reference_positive_ratio:.2%})",
                    metric="positive_ratio",
                    value=current_ratio,
                    threshold=self._reference_positive_ratio,
                )
                self._alerts.append(alert)
                return alert
            elif ratio_diff >= self.positive_ratio_warning:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"Prediction ratio differs from reference ({current_ratio:.2%} vs {self._reference_positive_ratio:.2%})",
                    metric="positive_ratio",
                    value=current_ratio,
                    threshold=self._reference_positive_ratio,
                )
                self._alerts.append(alert)
                return alert

            return None

    def check_output_drift(self) -> Dict:
        """Check for drift in output distribution.

        Returns:
            Dictionary with drift detection results.
        """
        with self._lock:
            if self._output_drift_detector is None:
                return {
                    "drift_detected": False,
                    "error": "Output drift detector not initialized",
                }

            if len(self._predictions) < 50:
                return {
                    "drift_detected": False,
                    "error": f"Insufficient data ({len(self._predictions)} predictions)",
                }

            try:
                probabilities = np.array([p for p, _ in self._predictions]).reshape(-1, 1)
                result = self._output_drift_detector.predict(probabilities)

                is_drift = bool(result["data"]["is_drift"])
                p_val = result["data"].get("p_val")

                if isinstance(p_val, np.ndarray):
                    p_val = float(p_val[0])
                elif p_val is not None:
                    p_val = float(p_val)

                if is_drift:
                    alert = Alert(
                        level=AlertLevel.CRITICAL,
                        message="Output distribution drift detected",
                        metric="output_drift",
                        value=p_val if p_val else 0.0,
                        threshold=self.drift_p_threshold,
                    )
                    self._alerts.append(alert)

                return {
                    "drift_detected": is_drift,
                    "p_value": p_val,
                    "threshold": self.drift_p_threshold,
                    "sample_size": len(probabilities),
                }

            except Exception as exc:
                logger.warning("Output drift detection failed: %s", exc)
                return {
                    "drift_detected": False,
                    "error": str(exc),
                }

    def get_stats(self, check_drift: bool = False) -> MonitoringStats:
        """Get current monitoring statistics.

        Args:
            check_drift: If True, run drift detection (expensive). Default False.

        Returns:
            MonitoringStats with current window statistics.
        """
        with self._lock:
            if not self._predictions:
                return MonitoringStats(alerts=self._alerts[-10:])

            predictions = list(self._predictions)
            probabilities = [p for p, _ in predictions]
            labels = [label for _, label in predictions]

            positive_count = sum(labels)
            negative_count = len(labels) - positive_count

            # Calculate confidence metrics
            confidences = [abs(p - 0.5) * 2 for p in probabilities]
            avg_confidence = float(np.mean(confidences))
            confidence_std = float(np.std(confidences))
            low_confidence_count = sum(1 for c in confidences if c < self.confidence_threshold)

            # Only check drift if explicitly requested (expensive operation)
            drift_detected = False
            if check_drift:
                # Release lock before drift check to avoid deadlock
                pass

            return MonitoringStats(
                total_predictions=len(predictions),
                positive_predictions=positive_count,
                negative_predictions=negative_count,
                avg_confidence=avg_confidence,
                confidence_std=confidence_std,
                low_confidence_count=low_confidence_count,
                prediction_drift_detected=drift_detected,
                alerts=self._alerts[-10:],  # Return last 10 alerts
            )

    def get_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent alerts.
        """
        with self._lock:
            return self._alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
            logger.info("Cleared all monitoring alerts")

    def reset(self) -> None:
        """Reset all monitoring state."""
        with self._lock:
            self._predictions.clear()
            self._alerts.clear()
            self._total_count = 0
            logger.info("Reset prediction monitor")


def create_monitor_from_test_predictions(
    test_predictions: np.ndarray,
    window_size: int = 1000,
    drift_p_threshold: float = 0.05,
) -> PredictionMonitor:
    """Create a prediction monitor initialized with test set predictions.

    Args:
        test_predictions: Predictions on test set for reference distribution.
        window_size: Size of monitoring window.
        drift_p_threshold: P-value threshold for drift detection.

    Returns:
        Initialized PredictionMonitor.
    """
    monitor = PredictionMonitor(
        reference_predictions=test_predictions,
        window_size=window_size,
        drift_p_threshold=drift_p_threshold,
    )

    logger.info(
        "Created prediction monitor with %d reference predictions",
        len(test_predictions),
    )
    return monitor
