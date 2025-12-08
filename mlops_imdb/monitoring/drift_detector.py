"""Drift detection module using alibi-detect for text data monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional

import joblib
import numpy as np
from alibi_detect.cd import ChiSquareDrift, KSDrift, TabularDrift
from alibi_detect.saving import load_detector, save_detector
from sklearn.feature_extraction.text import TfidfVectorizer

from mlops_imdb.logger import get_logger

logger = get_logger(__name__)


class DriftStatus(str, Enum):
    """Enumeration of possible drift detection statuses."""

    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


@dataclass
class DriftReport:
    """Report containing drift detection results."""

    status: DriftStatus
    p_value: Optional[float] = None
    threshold: float = 0.05
    drift_score: Optional[float] = None
    feature_drifts: Optional[dict] = None
    sample_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert report to dictionary for API responses."""
        return {
            "status": self.status.value,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_score": self.drift_score,
            "drift_detected": self.status == DriftStatus.DRIFT_DETECTED,
            "feature_drifts": self.feature_drifts,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class DriftDetector:
    """Drift detector for text data using alibi-detect.

    Supports multiple drift detection methods:
    - KS (Kolmogorov-Smirnov) test for continuous features
    - Chi-Square test for categorical features
    - Tabular drift for combined feature drift detection

    For text data, we first transform to TF-IDF features, then apply drift detection.
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
        p_threshold: float = 0.05,
        detector_type: str = "ks",
        min_samples: int = 50,
        window_size: int = 500,
    ):
        """Initialize the drift detector.

        Args:
            reference_data: Reference feature data (TF-IDF transformed) for drift comparison.
            vectorizer: Fitted TF-IDF vectorizer for text transformation.
            p_threshold: P-value threshold for drift detection.
            detector_type: Type of drift detector ('ks', 'chi2', 'tabular').
            min_samples: Minimum samples required before drift detection.
            window_size: Size of the sliding window for online detection.
        """
        self.vectorizer = vectorizer
        self.p_threshold = p_threshold
        self.detector_type = detector_type
        self.min_samples = min_samples
        self.window_size = window_size

        self._detector = None
        self._lock = Lock()
        self._sample_buffer: List[np.ndarray] = []
        self._initialized = False

        if reference_data is not None:
            self._initialize_detector(reference_data)

    def _initialize_detector(self, reference_data: np.ndarray) -> None:
        """Initialize the alibi-detect detector with reference data."""
        try:
            # Ensure reference data is dense and 2D
            if hasattr(reference_data, "toarray"):
                reference_data = reference_data.toarray()
            if reference_data.ndim == 1:
                reference_data = reference_data.reshape(-1, 1)

            # Select detector type
            if self.detector_type == "ks":
                self._detector = KSDrift(
                    reference_data,
                    p_val=self.p_threshold,
                )
            elif self.detector_type == "chi2":
                self._detector = ChiSquareDrift(
                    reference_data,
                    p_val=self.p_threshold,
                )
            elif self.detector_type == "tabular":
                self._detector = TabularDrift(
                    reference_data,
                    p_val=self.p_threshold,
                )
            else:
                # Default to KS drift
                self._detector = KSDrift(
                    reference_data,
                    p_val=self.p_threshold,
                )

            self._initialized = True
            logger.info(
                "Initialized %s drift detector with reference data shape: %s",
                self.detector_type,
                reference_data.shape,
            )
        except Exception as exc:
            logger.exception("Failed to initialize drift detector")
            raise RuntimeError(f"Failed to initialize drift detector: {exc}") from exc

    def transform_text(self, texts: List[str]) -> np.ndarray:
        """Transform text data to feature vectors using the vectorizer.

        Args:
            texts: List of text strings to transform.

        Returns:
            Feature matrix as numpy array.
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not set. Call set_vectorizer() first.")

        features = self.vectorizer.transform(texts)
        if hasattr(features, "toarray"):
            features = features.toarray()
        return features

    def set_vectorizer(self, vectorizer: TfidfVectorizer) -> None:
        """Set the TF-IDF vectorizer for text transformation."""
        self.vectorizer = vectorizer
        logger.info("Vectorizer set for drift detector")

    def set_reference_data(
        self,
        reference_texts: Optional[List[str]] = None,
        reference_features: Optional[np.ndarray] = None,
    ) -> None:
        """Set reference data for drift detection.

        Args:
            reference_texts: Reference texts (will be transformed using vectorizer).
            reference_features: Pre-computed reference features.
        """
        with self._lock:
            if reference_features is not None:
                self._initialize_detector(reference_features)
            elif reference_texts is not None:
                if self.vectorizer is None:
                    raise RuntimeError("Vectorizer required to transform reference texts")
                features = self.transform_text(reference_texts)
                self._initialize_detector(features)
            else:
                raise ValueError("Either reference_texts or reference_features must be provided")

    def add_sample(self, text: str) -> None:
        """Add a single sample to the buffer for batch drift detection.

        Args:
            text: Input text to add to buffer.
        """
        if self.vectorizer is None:
            logger.warning("Vectorizer not set, cannot add sample")
            return

        with self._lock:
            try:
                features = self.transform_text([text])
                self._sample_buffer.append(features[0])

                # Keep buffer within window size
                if len(self._sample_buffer) > self.window_size:
                    self._sample_buffer = self._sample_buffer[-self.window_size :]
            except Exception as exc:
                logger.warning("Failed to add sample to buffer: %s", exc)

    def add_samples_batch(self, texts: List[str]) -> None:
        """Add multiple samples to the buffer.

        Args:
            texts: List of input texts.
        """
        if self.vectorizer is None:
            logger.warning("Vectorizer not set, cannot add samples")
            return

        with self._lock:
            try:
                features = self.transform_text(texts)
                for feature in features:
                    self._sample_buffer.append(feature)

                # Keep buffer within window size
                if len(self._sample_buffer) > self.window_size:
                    self._sample_buffer = self._sample_buffer[-self.window_size :]
            except Exception as exc:
                logger.warning("Failed to add samples to buffer: %s", exc)

    def detect_drift(
        self,
        texts: Optional[List[str]] = None,
        features: Optional[np.ndarray] = None,
        use_buffer: bool = False,
    ) -> DriftReport:
        """Detect drift in the provided data or buffer.

        Args:
            texts: Text data to check for drift.
            features: Pre-computed features to check for drift.
            use_buffer: If True, use accumulated buffer for detection.

        Returns:
            DriftReport with detection results.
        """
        with self._lock:
            if not self._initialized:
                return DriftReport(
                    status=DriftStatus.ERROR,
                    details={"error": "Detector not initialized with reference data"},
                )

            try:
                # Determine data source
                if use_buffer:
                    if len(self._sample_buffer) < self.min_samples:
                        return DriftReport(
                            status=DriftStatus.INSUFFICIENT_DATA,
                            sample_size=len(self._sample_buffer),
                            details={
                                "message": f"Need at least {self.min_samples} samples, have {len(self._sample_buffer)}"
                            },
                        )
                    test_data = np.array(self._sample_buffer)
                elif features is not None:
                    test_data = features
                    if hasattr(test_data, "toarray"):
                        test_data = test_data.toarray()
                elif texts is not None:
                    if len(texts) < self.min_samples:
                        return DriftReport(
                            status=DriftStatus.INSUFFICIENT_DATA,
                            sample_size=len(texts),
                            details={
                                "message": f"Need at least {self.min_samples} samples, have {len(texts)}"
                            },
                        )
                    test_data = self.transform_text(texts)
                else:
                    return DriftReport(
                        status=DriftStatus.ERROR,
                        details={"error": "No data provided for drift detection"},
                    )

                if test_data.ndim == 1:
                    test_data = test_data.reshape(-1, 1)

                # Run drift detection
                result = self._detector.predict(test_data)

                # Extract results
                is_drift = bool(result["data"]["is_drift"])
                p_val = result["data"].get("p_val")

                # Handle p_val which might be an array
                # For multi-dimensional features, alibi-detect returns p-values per feature
                # We use the minimum for drift decision, but report mean for better interpretability
                if isinstance(p_val, np.ndarray):
                    p_val_min = float(np.min(p_val))
                    p_val_mean = float(np.mean(p_val))
                    # Use mean for reporting (more stable), but log min for debugging
                    p_val_scalar = p_val_mean
                    logger.debug(
                        "P-value array: min=%.6f, mean=%.6f, shape=%s",
                        p_val_min,
                        p_val_mean,
                        p_val.shape,
                    )
                    # If min is 0.0 but mean is reasonable, it's likely numerical precision
                    if p_val_min == 0.0 and p_val_mean > 0.0:
                        logger.debug(
                            "P-value min is 0.0 but mean is %.6f - using mean for reporting",
                            p_val_mean,
                        )
                elif p_val is not None:
                    p_val_scalar = float(p_val)
                else:
                    p_val_scalar = None

                # Calculate drift score
                distance = result["data"].get("distance")
                if isinstance(distance, np.ndarray):
                    drift_score = float(np.mean(distance))
                elif distance is not None:
                    drift_score = float(distance)
                else:
                    drift_score = None

                # Feature-level drift information
                feature_drifts = None
                if "is_drift" in result["data"] and isinstance(
                    result["data"]["is_drift"], np.ndarray
                ):
                    feature_drifts = {
                        "num_drifted_features": int(np.sum(result["data"]["is_drift"])),
                        "total_features": len(result["data"]["is_drift"]),
                    }

                status = DriftStatus.DRIFT_DETECTED if is_drift else DriftStatus.NO_DRIFT

                # Build details with additional p-value information
                details = {"detector_type": self.detector_type}
                if isinstance(p_val, np.ndarray):
                    details["p_value_min"] = float(np.min(p_val))
                    details["p_value_mean"] = float(np.mean(p_val))
                    details["p_value_median"] = float(np.median(p_val))
                    details["p_value_std"] = float(np.std(p_val))

                return DriftReport(
                    status=status,
                    p_value=p_val_scalar,
                    threshold=self.p_threshold,
                    drift_score=drift_score,
                    feature_drifts=feature_drifts,
                    sample_size=len(test_data),
                    details=details,
                )

            except Exception as exc:
                logger.exception("Drift detection failed")
                return DriftReport(
                    status=DriftStatus.ERROR,
                    details={"error": str(exc)},
                )

    def clear_buffer(self) -> None:
        """Clear the sample buffer."""
        with self._lock:
            self._sample_buffer.clear()
            logger.info("Drift detector buffer cleared")

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._sample_buffer)

    def save(self, path: str) -> None:
        """Save the drift detector to disk.

        Args:
            path: Directory path to save the detector.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._detector is not None:
            save_detector(self._detector, str(save_path / "alibi_detector"))

        # Save vectorizer and config
        config = {
            "p_threshold": self.p_threshold,
            "detector_type": self.detector_type,
            "min_samples": self.min_samples,
            "window_size": self.window_size,
            "initialized": self._initialized,
        }
        joblib.dump(config, save_path / "config.pkl")

        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, save_path / "vectorizer.pkl")

        logger.info("Saved drift detector to %s", path)

    @classmethod
    def load(cls, path: str) -> "DriftDetector":
        """Load a drift detector from disk.

        Args:
            path: Directory path containing the saved detector.

        Returns:
            Loaded DriftDetector instance.
        """
        load_path = Path(path)

        config = joblib.load(load_path / "config.pkl")

        vectorizer = None
        vectorizer_path = load_path / "vectorizer.pkl"
        if vectorizer_path.exists():
            vectorizer = joblib.load(vectorizer_path)

        detector = cls(
            vectorizer=vectorizer,
            p_threshold=config["p_threshold"],
            detector_type=config["detector_type"],
            min_samples=config["min_samples"],
            window_size=config["window_size"],
        )

        alibi_path = load_path / "alibi_detector"
        if alibi_path.exists() and config.get("initialized", False):
            detector._detector = load_detector(str(alibi_path))
            detector._initialized = True

        logger.info("Loaded drift detector from %s", path)
        return detector


def create_detector_from_training_data(
    train_texts: List[str],
    vectorizer: TfidfVectorizer,
    p_threshold: float = 0.05,
    detector_type: str = "ks",
    sample_size: Optional[int] = None,
) -> DriftDetector:
    """Create a drift detector initialized with training data.

    Args:
        train_texts: Training text data for reference distribution.
        vectorizer: Fitted TF-IDF vectorizer.
        p_threshold: P-value threshold for drift detection.
        detector_type: Type of detector ('ks', 'chi2', 'tabular').
        sample_size: Optional sample size (uses subset if specified).

    Returns:
        Initialized DriftDetector instance.
    """
    # Sample if needed
    if sample_size is not None and len(train_texts) > sample_size:
        indices = np.random.choice(len(train_texts), sample_size, replace=False)
        train_texts = [train_texts[i] for i in indices]

    # Transform to features
    reference_features = vectorizer.transform(train_texts)
    if hasattr(reference_features, "toarray"):
        reference_features = reference_features.toarray()

    detector = DriftDetector(
        reference_data=reference_features,
        vectorizer=vectorizer,
        p_threshold=p_threshold,
        detector_type=detector_type,
    )

    logger.info(
        "Created drift detector with %d reference samples",
        len(train_texts),
    )
    return detector

