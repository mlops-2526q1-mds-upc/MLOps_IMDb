"""Utility functions for initializing and managing monitoring components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from mlops_imdb.config import DATA_DIR, MODELS_DIR
from mlops_imdb.logger import get_logger
from mlops_imdb.monitoring.drift_detector import DriftDetector
from mlops_imdb.monitoring.prediction_monitor import PredictionMonitor

logger = get_logger(__name__)

# Default paths
DEFAULT_DETECTOR_PATH = MODELS_DIR / "drift_detector"
DEFAULT_MONITOR_PATH = MODELS_DIR / "prediction_monitor"


def load_params(path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def initialize_sentiment_monitoring(
    vectorizer_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    sample_size: int = 1000,
    p_threshold: float = 0.05,
) -> Tuple[Optional[DriftDetector], Optional[PredictionMonitor]]:
    """Initialize monitoring components for sentiment analysis.

    Args:
        vectorizer_path: Path to the TF-IDF vectorizer.
        train_data_path: Path to training data CSV.
        model_path: Path to the trained model.
        sample_size: Number of samples for reference data.
        p_threshold: P-value threshold for drift detection.

    Returns:
        Tuple of (DriftDetector, PredictionMonitor) or (None, None) on failure.
    """
    params = load_params()

    # Resolve paths from params if not provided
    if vectorizer_path is None:
        vectorizer_path = params.get("features", {}).get("outputs", {}).get(
            "vectorizer_path", str(MODELS_DIR / "tfidf_vectorizer.pkl")
        )
    if train_data_path is None:
        train_data_path = params.get("data", {}).get("processed", {}).get(
            "train", str(DATA_DIR / "processed" / "imdb_train_clean.csv")
        )
    if model_path is None:
        model_path = params.get("train", {}).get("outputs", {}).get(
            "model_path", str(MODELS_DIR / "model.pkl")
        )

    drift_detector = None
    prediction_monitor = None

    try:
        # Load vectorizer
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            logger.info("Loaded vectorizer from %s", vectorizer_path)

            # Load training data for reference
            if os.path.exists(train_data_path):
                df_train = pd.read_csv(train_data_path)
                text_col = params.get("data", {}).get("schema", {}).get("text_col", "text")

                if text_col in df_train.columns:
                    texts = df_train[text_col].tolist()

                    # Sample if needed
                    if len(texts) > sample_size:
                        indices = np.random.choice(len(texts), sample_size, replace=False)
                        texts = [texts[i] for i in indices]

                    # Transform to features
                    reference_features = vectorizer.transform(texts)
                    if hasattr(reference_features, "toarray"):
                        reference_features = reference_features.toarray()

                    # Create drift detector
                    drift_detector = DriftDetector(
                        reference_data=reference_features,
                        vectorizer=vectorizer,
                        p_threshold=p_threshold,
                        detector_type="ks",
                    )
                    logger.info(
                        "Created drift detector with %d reference samples", len(texts)
                    )
        else:
            logger.warning("Vectorizer not found at %s", vectorizer_path)

        # Load model and generate reference predictions for output monitor
        if os.path.exists(model_path) and drift_detector is not None:
            model = joblib.load(model_path)

            # Load test data for reference predictions
            test_data_path = params.get("data", {}).get("processed", {}).get(
                "test", str(DATA_DIR / "processed" / "imdb_test_clean.csv")
            )

            if os.path.exists(test_data_path):
                df_test = pd.read_csv(test_data_path)
                text_col = params.get("data", {}).get("schema", {}).get("text_col", "text")

                if text_col in df_test.columns:
                    test_texts = df_test[text_col].tolist()[:sample_size]
                    test_features = vectorizer.transform(test_texts)

                    try:
                        test_predictions = model.predict_proba(test_features)[:, 1]
                    except AttributeError:
                        # Model doesn't have predict_proba, use predict
                        test_predictions = model.predict(test_features)
                        test_predictions = np.array(test_predictions, dtype=float)

                    prediction_monitor = PredictionMonitor(
                        reference_predictions=test_predictions,
                        drift_p_threshold=p_threshold,
                    )
                    logger.info(
                        "Created prediction monitor with %d reference predictions",
                        len(test_predictions),
                    )

    except Exception as exc:
        logger.exception("Failed to initialize sentiment monitoring: %s", exc)

    return drift_detector, prediction_monitor


def initialize_spam_monitoring(
    vocab_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    sample_size: int = 1000,
    p_threshold: float = 0.05,
) -> Tuple[Optional[DriftDetector], Optional[PredictionMonitor]]:
    """Initialize monitoring components for spam detection.

    For the spam model, we use token-based features rather than TF-IDF.

    Args:
        vocab_path: Path to vocabulary JSON.
        train_data_path: Path to training tensor data.
        sample_size: Number of samples for reference data.
        p_threshold: P-value threshold for drift detection.

    Returns:
        Tuple of (DriftDetector, PredictionMonitor) or (None, None) on failure.
    """
    params = load_params()
    spam_cfg = params.get("spam", {})

    # For spam, we'll use a simpler approach - just prediction monitoring
    # since the model uses embeddings rather than TF-IDF
    prediction_monitor = None

    try:
        # Create a prediction monitor with default reference
        # It will be updated with actual predictions over time
        prediction_monitor = PredictionMonitor(
            window_size=1000,
            drift_p_threshold=p_threshold,
        )
        logger.info("Created spam prediction monitor")

    except Exception as exc:
        logger.exception("Failed to initialize spam monitoring: %s", exc)

    return None, prediction_monitor


def save_monitoring_state(
    drift_detector: Optional[DriftDetector] = None,
    prediction_monitor: Optional[PredictionMonitor] = None,
    detector_path: Optional[str] = None,
    monitor_path: Optional[str] = None,
) -> None:
    """Save monitoring component state to disk.

    Args:
        drift_detector: DriftDetector instance to save.
        prediction_monitor: PredictionMonitor instance (stats only).
        detector_path: Path to save drift detector.
        monitor_path: Path to save monitor state.
    """
    if drift_detector is not None:
        path = detector_path or str(DEFAULT_DETECTOR_PATH)
        drift_detector.save(path)
        logger.info("Saved drift detector to %s", path)

    if prediction_monitor is not None:
        path = monitor_path or str(DEFAULT_MONITOR_PATH)
        Path(path).mkdir(parents=True, exist_ok=True)
        stats = prediction_monitor.get_stats()
        joblib.dump(stats.to_dict(), Path(path) / "stats.pkl")
        logger.info("Saved prediction monitor stats to %s", path)


def load_drift_detector(path: Optional[str] = None) -> Optional[DriftDetector]:
    """Load a drift detector from disk.

    Args:
        path: Path to the saved detector.

    Returns:
        Loaded DriftDetector or None if not found.
    """
    load_path = path or str(DEFAULT_DETECTOR_PATH)

    if not os.path.exists(load_path):
        logger.warning("Drift detector not found at %s", load_path)
        return None

    try:
        return DriftDetector.load(load_path)
    except Exception as exc:
        logger.exception("Failed to load drift detector: %s", exc)
        return None


def get_monitoring_config() -> dict:
    """Get monitoring configuration from params.yaml.

    Returns:
        Dictionary with monitoring configuration.
    """
    params = load_params()
    monitoring_cfg = params.get("monitoring", {})

    # Provide defaults
    defaults = {
        "enabled": True,
        "drift_detection": {
            "enabled": True,
            "p_threshold": 0.05,
            "detector_type": "ks",
            "min_samples": 50,
            "window_size": 500,
            "sample_size": 1000,
        },
        "prediction_monitoring": {
            "enabled": True,
            "window_size": 1000,
            "confidence_threshold": 0.6,
            "positive_ratio_warning": 0.3,
            "positive_ratio_critical": 0.2,
        },
        "alerting": {
            "enabled": True,
            "log_alerts": True,
        },
    }

    # Merge with loaded config
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(defaults, monitoring_cfg)

