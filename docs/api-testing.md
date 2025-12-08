# API Testing Documentation

This document describes the FastAPI endpoint tests for both the Sentiment API and Spam API. These tests ensure that the production APIs behave correctly, handle edge cases, and return expected response formats.

## Overview

**Test Files:**
- `tests/modeling/test_api.py` - Sentiment API tests (11 tests)
- `tests/spam/test_api.py` - Spam API tests (6 tests)

**Test Statistics:**
- Total API tests: 17
- Sentiment API: 11 tests (1 health, 10 predict)
- Spam API: 6 tests (1 health, 5 predict)

**Test Framework:**
- FastAPI TestClient for HTTP endpoint testing
- pytest for test organization and execution
- Mock models to avoid dependency on actual trained models

**Status:** All API tests pass ✓

## Test Architecture

### Mocking Strategy

Both test suites use mock models to:
- Avoid dependency on trained model files
- Enable fast test execution
- Provide predictable test behavior
- Test API logic independently of model quality

**Mock Model Behavior:**
- Sentiment API: Returns probabilities based on keyword matching (e.g., "great" → 0.92, "terrible" → 0.12)
- Spam API: Returns probabilities based on spam keywords (e.g., "spam" or "buy now" → 0.95, otherwise → 0.15)

## Sentiment API Tests (`tests/modeling/test_api.py`)

### Test Coverage

#### 1. Health Endpoint (`/health`)

**Test:** `test_health_returns_ok`

**What We Check:**
- HTTP status code is 200
- Response contains `"status": "ok"`
- Response includes `model_uri` field

**Why This Test:**
- Health checks are critical for monitoring and load balancers
- Ensures API is ready to serve requests
- Validates response structure for automated health checks

**Status:** ✓ Passes

#### 2. Predict Endpoint (`/predict`)

**Test Suite:** `TestPredictEndpoint`

**What We Check:**

**a) Positive Sentiment Classification**
- Test: `test_predict_positive_sentiment`
- Verifies correct classification of positive text
- Validates response contains: `probability`, `label`, `sentiment`
- Ensures `label == 1` and `sentiment == "positive"`
- Verifies `probability >= 0.5`

**b) Negative Sentiment Classification**
- Test: `test_predict_negative_sentiment`
- Verifies correct classification of negative text
- Ensures `label == 0` and `sentiment == "negative"`
- Verifies `probability < 0.5`

**c) Probability Range Validation**
- Test: `test_predict_returns_probability_in_range`
- Ensures probability is between 0.0 and 1.0
- Critical for downstream processing and confidence thresholds

**d) Label Validation**
- Test: `test_predict_returns_valid_label`
- Ensures label is binary: either 0 or 1
- Prevents invalid classification outputs

**e) Sentiment String Validation**
- Test: `test_predict_returns_valid_sentiment`
- Ensures sentiment is either "positive" or "negative"
- Validates string format for UI display

**f) Missing Text Field Handling**
- Test: `test_predict_missing_text_field`
- Verifies 422 status code for missing required field
- Validates FastAPI request validation

**g) Invalid JSON Handling**
- Test: `test_predict_invalid_json`
- Verifies 422 status code for malformed JSON
- Ensures API gracefully handles bad requests

**h) Empty Text Handling**
- Test: `test_predict_empty_text`
- Verifies API handles empty strings without crashing
- Returns valid response structure

**i) Long Text Handling**
- Test: `test_predict_long_text`
- Verifies API processes very long input text
- Ensures no truncation or memory issues

**j) Special Characters Handling**
- Test: `test_predict_special_characters`
- Verifies API handles HTML entities, special chars, symbols
- Ensures robust text preprocessing

**Why These Tests:**
- **Correctness:** Ensure predictions match expected sentiment
- **Robustness:** Handle edge cases gracefully
- **Validation:** Enforce response format consistency
- **Security:** Prevent crashes from malicious input
- **User Experience:** Handle various input formats

**Status:** All tests pass ✓

## Spam API Tests (`tests/spam/test_api.py`)

### Test Coverage

#### 1. Health Endpoint (`/health`)

**Test:** `test_health_returns_ok`

**What We Check:**
- HTTP status code is 200
- Response contains `"status": "ok"`
- Response includes `model_uri` field

**Why This Test:**
- Consistent health check behavior across APIs
- Enables unified monitoring
- Validates API readiness

**Status:** ✓ Passes

#### 2. Predict Endpoint (`/predict`)

**Test Suite:** `TestPredictEndpoint`

**What We Check:**

**a) Spam Text Classification**
- Test: `test_predict_spam_text`
- Verifies correct classification of spam-like text
- Validates response contains: `probability`, `label`
- Ensures `label == 1` for spam
- Verifies `probability >= 0.5` for spam

**b) Ham (Non-Spam) Text Classification**
- Test: `test_predict_ham_text`
- Verifies correct classification of normal text
- Ensures `label == 0` for non-spam
- Verifies `probability < 0.5` for non-spam

**c) Probability Range Validation**
- Test: `test_predict_returns_probability_in_range`
- Ensures probability is between 0.0 and 1.0
- Critical for threshold-based filtering

**d) Missing Text Field Handling**
- Test: `test_predict_missing_text_field`
- Verifies 422 status code for missing required field
- Validates FastAPI request validation

**e) Invalid JSON Handling**
- Test: `test_predict_invalid_json`
- Verifies 422 status code for malformed JSON
- Ensures API gracefully handles bad requests

**f) Empty Text Handling**
- Test: `test_predict_empty_text`
- Verifies API handles empty strings without crashing
- Returns valid response structure

**Why These Tests:**
- **Binary Classification:** Ensure correct spam/ham distinction
- **Probability Accuracy:** Validate confidence scores
- **Error Handling:** Graceful degradation on bad input
- **API Consistency:** Same behavior patterns as Sentiment API

**Status:** All tests pass ✓

## Test Execution

### Running API Tests

```powershell
# Run all API tests
uv run pytest tests/modeling/test_api.py tests/spam/test_api.py

# Run Sentiment API tests only
uv run pytest tests/modeling/test_api.py

# Run Spam API tests only
uv run pytest tests/spam/test_api.py

# Run with verbose output
uv run pytest tests/modeling/test_api.py -v

# Run specific test class
uv run pytest tests/modeling/test_api.py::TestPredictEndpoint

# Run specific test
uv run pytest tests/modeling/test_api.py::TestPredictEndpoint::test_predict_positive_sentiment
```

### Test Output Example

```
tests/modeling/test_api.py::TestHealthEndpoint::test_health_returns_ok PASSED
tests/modeling/test_api.py::TestPredictEndpoint::test_predict_positive_sentiment PASSED
tests/modeling/test_api.py::TestPredictEndpoint::test_predict_negative_sentiment PASSED
...
tests/spam/test_api.py::TestHealthEndpoint::test_health_returns_ok PASSED
tests/spam/test_api.py::TestPredictEndpoint::test_predict_spam_text PASSED
tests/spam/test_api.py::TestPredictEndpoint::test_predict_ham_text PASSED
...
```

## Test Design Decisions

### 1. Why Mock Models?

**Decision:** Use mock models instead of loading real trained models

**Reasons:**
- **Speed:** Tests run in milliseconds instead of seconds
- **Independence:** Tests don't require model files or MLflow setup
- **Predictability:** Mock behavior is deterministic
- **Focus:** Tests API logic, not model quality
- **CI/CD:** No need to download large model files in CI

**Trade-off:** We don't test actual model predictions, but that's covered in evaluation tests.

### 2. Why Test Edge Cases?

**Decision:** Include tests for empty text, long text, special characters, invalid JSON

**Reasons:**
- **Production Reality:** Users send unexpected input
- **Security:** Prevent crashes from malicious input
- **Robustness:** API must handle all input gracefully
- **User Experience:** Better error messages improve UX

### 3. Why Test Response Structure?

**Decision:** Validate specific fields in responses (probability, label, sentiment)

**Reasons:**
- **API Contract:** Clients depend on consistent response format
- **Integration:** Frontend and monitoring systems parse these fields
- **Debugging:** Consistent structure aids troubleshooting
- **Documentation:** Tests serve as API specification

### 4. Why Separate Test Files?

**Decision:** Separate test files for Sentiment and Spam APIs

**Reasons:**
- **Clarity:** Each API has different behavior
- **Maintainability:** Easier to find and update tests
- **Selective Execution:** Can test one API independently
- **Organization:** Matches code structure

## Coverage Gaps and Future Tests

### Currently Not Tested (But Could Be)

1. **Monitoring Endpoints**
   - `/monitoring/drift` (sentiment only)
   - `/monitoring/stats`
   - `/monitoring/alerts`
   - `/monitoring/reset`
   - `/monitoring/reinitialize`

2. **Internal Admin Endpoints**
   - `/reload` (model reload)

3. **Concurrent Requests**
   - Load testing
   - Thread safety

4. **Authentication/Authorization**
   - If added in future

5. **Rate Limiting**
   - If added in future

### Why These Aren't Tested Yet

- Monitoring endpoints require initialized monitoring components
- Internal endpoints require model files
- Load testing requires separate performance test suite
- Authentication not yet implemented

## Integration with CI/CD

API tests are run automatically:
- On every pull request
- Before merging to main branch
- As part of deployment pipeline

**Benefits:**
- Catch breaking changes immediately
- Ensure API compatibility
- Validate before deployment
- Maintain code quality

## Best Practices Demonstrated

1. **Test Isolation:** Each test is independent
2. **Clear Naming:** Test names describe what they test
3. **Comprehensive Coverage:** Happy paths and error cases
4. **Fast Execution:** Mocking enables quick tests
5. **Maintainable:** Easy to add new tests
6. **Documented:** Tests serve as API documentation

## Conclusion

The API test suite provides confidence that:
- ✅ Endpoints return correct HTTP status codes
- ✅ Response formats are consistent
- ✅ Edge cases are handled gracefully
- ✅ Input validation works correctly
- ✅ APIs are ready for production use

All tests pass, ensuring reliable API behavior for both sentiment and spam prediction services.

