# Test Suite Documentation

This document describes the comprehensive test suite for the MLOps IMDb project, covering all components except API endpoints (which are documented separately in `api-testing.md`).

## Test Overview

The test suite ensures reliability, correctness, and maintainability across all project components. All tests pass and are run automatically via pytest.

**Test Statistics:**
- Total tests: 83
- Total test files: 14
- Test categories: Data preparation, feature engineering, model training, evaluation, monitoring, API endpoints
- Framework: pytest with fixtures and mocking

**Test Breakdown by Category:**
- CodeCarbon tracking: 12 tests
- Data preparation: 6 tests
- Feature engineering: 1 test
- Sentiment modeling (API, train, eval): 16 tests
- Spam modeling (API, train, eval, features, model, prepare): 14 tests
- Monitoring (drift detection, prediction monitoring): 28 tests
- API endpoints (sentiment + spam): 17 tests (documented separately in `api-testing.md`)

## Test Categories

### 1. Data Preparation Tests (`tests/data/test_prepare.py`)

**Purpose:** Verify that raw data is correctly cleaned and preprocessed before model training.

**What We Check:**
- Text cleaning functions (lowercase, HTML tag removal, whitespace normalization)
- Configuration flag respect (disabled preprocessing steps)
- HTML entity decoding
- End-to-end CSV processing pipeline
- Missing column validation
- YAML parameter loading

**Why These Tests:**
- Data quality is critical for model performance
- Preprocessing bugs can silently degrade model accuracy
- Configuration flexibility must be preserved
- Input validation prevents runtime errors

**Key Tests:**
- `test_clean_text_applies_preprocessing`: Verifies all cleaning steps work together
- `test_clean_text_respects_disabled_flags`: Ensures configuration is respected
- `test_main_processes_raw_csvs`: End-to-end pipeline validation
- `test_main_raises_for_missing_columns`: Input validation

**Status:** All tests pass ✓

### 2. Feature Engineering Tests (`tests/features/test_build_features.py`)

**Purpose:** Validate TF-IDF feature extraction and vectorizer persistence.

**What We Check:**
- TF-IDF feature matrix generation from text data
- Sparse matrix format (CSR) correctness
- Vectorizer persistence (pickle serialization)
- Train/test feature consistency
- Configuration parameter application (max_features, ngram_range, etc.)

**Why These Tests:**
- Feature extraction is the foundation of model input
- Vectorizer must be saved for inference
- Sparse matrices must be correctly formatted for scikit-learn
- Configuration must be correctly applied

**Key Tests:**
- `test_main_builds_tfidf_features`: Complete feature extraction pipeline
- Verifies output file existence and format
- Validates feature matrix dimensions

**Status:** All tests pass ✓

### 3. Model Training Tests

#### Sentiment Model (`tests/modeling/test_train.py`)

**Purpose:** Verify sentiment model training pipeline.

**What We Check:**
- Model training completes successfully
- Model is saved to disk
- Trained model has expected attributes (coefficients)
- Model predictions are valid (binary labels: 0 or 1)
- MLflow tracking integration
- Parameter loading from config

**Why These Tests:**
- Training pipeline must produce valid models
- Model persistence is required for deployment
- MLflow integration enables experiment tracking
- Configuration-driven training ensures reproducibility

**Key Tests:**
- `test_main_trains_and_saves_model`: Complete training pipeline with mocked MLflow

**Status:** All tests pass ✓

#### Spam Model (`tests/spam/test_train.py`)

**Purpose:** Verify spam model (PyTorch) training pipeline.

**What We Check:**
- PyTorch model training with tokenized inputs
- Model checkpoint saving
- Training loss curve generation
- Vocabulary persistence
- Configuration parameter application (embedding_dim, hidden_dim, epochs, etc.)

**Why These Tests:**
- Deep learning models require different validation than scikit-learn
- Training curves help diagnose overfitting
- Vocabulary must be saved for inference
- Configuration must control all hyperparameters

**Key Tests:**
- `test_main_trains_and_saves_model`: Complete PyTorch training pipeline

**Status:** All tests pass ✓

### 4. Model Evaluation Tests

#### Sentiment Model (`tests/modeling/test_eval.py`)

**Purpose:** Validate model evaluation metrics and visualization generation.

**What We Check:**
- Metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix generation and PNG export
- MLflow metric logging
- Directory creation for outputs
- PNG file format validation

**Why These Tests:**
- Evaluation metrics must be accurate for model selection
- Visualizations aid in model understanding
- MLflow logging enables experiment comparison
- Output directory structure must be created automatically

**Key Tests:**
- `test_ensure_dir_creates_parent`: Directory creation utility
- `test_save_confusion_matrix_png`: Visualization generation
- `test_main_writes_metrics_and_confusion_matrix`: Complete evaluation pipeline

**Status:** All tests pass ✓

#### Spam Model (`tests/spam/test_eval.py`)

**Purpose:** Validate spam model evaluation pipeline.

**What We Check:**
- PyTorch model evaluation
- Metrics calculation for binary classification
- MLflow integration
- Output file generation

**Why These Tests:**
- Consistent evaluation across both models
- MLflow enables experiment tracking
- Metrics must be accurate for model comparison

**Status:** All tests pass ✓

### 5. Monitoring Tests

#### Drift Detection (`tests/monitoring/test_drift_detector.py`)

**Purpose:** Validate alibi-detect drift detection implementation.

**What We Check:**
- Detector initialization with reference data
- Detector initialization without reference (lazy initialization)
- Reference data setting after initialization
- Sample buffer management (add_sample, add_samples_batch)
- Window size limits (sliding window behavior)
- Drift detection with insufficient data
- Drift detection with similar data (no drift expected)
- Drift detection with different data (drift expected)
- Error handling when detector not initialized
- Buffer clearing functionality
- DriftReport serialization (to_dict)
- Factory function for creating detectors from training data

**Why These Tests:**
- Drift detection is critical for production monitoring
- Buffer management affects memory usage
- Window size limits prevent unbounded memory growth
- Error handling ensures graceful degradation
- Serialization enables API responses

**Key Tests:**
- `test_initialization_with_reference_data`: Core initialization
- `test_buffer_window_limit`: Memory management
- `test_detect_drift_insufficient_data`: Minimum sample validation
- `test_detect_drift_with_drift`: Actual drift detection logic
- `test_drift_report_to_dict`: API response format

**Status:** All tests pass ✓

#### Prediction Monitoring (`tests/monitoring/test_prediction_monitor.py`)

**Purpose:** Validate prediction distribution monitoring and alerting.

**What We Check:**
- Monitor initialization with/without reference predictions
- Prediction recording and statistics tracking
- Window size limits for prediction history
- Prediction ratio drift detection
- Output drift detection (KS test on predictions)
- Statistics retrieval (total, positive, negative, confidence)
- Alert generation and retrieval
- Alert clearing
- Monitor reset functionality
- Alert serialization (to_dict)
- MonitoringStats serialization
- Factory function for creating monitors from test predictions

**Why These Tests:**
- Prediction monitoring detects model degradation
- Ratio drift indicates data distribution shifts
- Output drift detects prediction distribution changes
- Alerts enable proactive model maintenance
- Statistics enable dashboard visualization

**Key Tests:**
- `test_record_prediction_stats`: Statistics tracking
- `test_window_size_limit`: Memory management
- `test_check_prediction_ratio_drift`: Ratio drift detection
- `test_check_output_drift`: Output drift detection
- `test_get_stats`: Statistics retrieval
- `test_clear_alerts`: Alert management

**Status:** All tests pass ✓

### 6. CodeCarbon Tracking Tests (`tests/codecarbon/test_trackers.py`)

**Purpose:** Validate energy consumption tracking integration.

**What We Check:**
- CodeCarbon tracker initialization
- Energy tracking during model training
- Carbon footprint calculation
- Report generation

**Why These Tests:**
- Environmental impact tracking is important for responsible AI
- Energy consumption affects cloud costs
- Tracking enables optimization opportunities

**Status:** All tests pass ✓

## Test Execution

### Running All Tests

```powershell
# Run all tests
uv run pytest tests

# Run with verbose output
uv run pytest tests -v

# Run specific test category
uv run pytest tests/monitoring/
uv run pytest tests/modeling/
uv run pytest tests/spam/
```

### Running Specific Tests

```powershell
# Run specific test file
uv run pytest tests/monitoring/test_drift_detector.py

# Run specific test class
uv run pytest tests/monitoring/test_drift_detector.py::TestDriftDetector

# Run specific test method
uv run pytest tests/monitoring/test_drift_detector.py::TestDriftDetector::test_detect_drift_with_drift
```

### Test Coverage

To generate coverage reports:

```powershell
uv run pytest tests --cov=mlops_imdb --cov-report=html
```

## Test Design Principles

### 1. Isolation
- Each test is independent and can run in any order
- Tests use fixtures for setup/teardown
- Mocked dependencies prevent external system dependencies

### 2. Clarity
- Test names clearly describe what is being tested
- Tests follow Arrange-Act-Assert pattern
- Assertions are specific and meaningful

### 3. Maintainability
- Fixtures reduce code duplication
- Mocking enables fast test execution
- Configuration is externalized via monkeypatching

### 4. Coverage
- Tests cover happy paths and error cases
- Edge cases are explicitly tested
- Integration tests validate end-to-end workflows

## Why These Tests Matter

1. **Reliability:** Catch bugs before deployment
2. **Documentation:** Tests serve as executable specifications
3. **Refactoring Safety:** Tests enable confident code changes
4. **Regression Prevention:** Tests catch breaking changes
5. **Design Validation:** Tests ensure components work as designed

## Continuous Integration

All tests are run automatically in CI/CD pipelines to ensure:
- Code quality is maintained
- Breaking changes are caught early
- All components integrate correctly
- Monitoring functionality works as expected

## Future Test Additions

Potential areas for additional testing:
- End-to-end integration tests with real data
- Performance benchmarks
- Load testing for APIs
- Model serving latency tests
- Monitoring alert delivery tests

