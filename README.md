# MLOps_IMDb

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Film review sentiment analysis.

## Setup Instructions

Follow these steps to set up the project environment and run the complete pipeline:

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [DVC](https://dvc.org/) for data versioning and pipeline management

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLOps_IMDb
   ```

2. **Create and activate virtual environment**
   ```bash
   make create_environment
   source ./.venv/bin/activate  # On Unix/macOS
   # or
   .\.venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   make requirements
   # or directly with uv
   uv sync
   ```

4. **Run the complete DVC pipeline**
   ```bash
   dvc pull
   dvc repro
   ```

This will execute all pipeline stages in order:
- **prepare**: Clean and preprocess the raw IMDB data
- **features**: Build TF-IDF features from the cleaned text
- **train**: Train the sentiment analysis model
- **eval**: Evaluate the model and generate metrics

### Alternative: Run individual stages

You can also run individual pipeline stages:

```bash
# Data preparation
dvc repro prepare

# Feature engineering
dvc repro features

# Model training
dvc repro train

# Model evaluation
dvc repro eval
```

### Verify Setup

After running the pipeline, you should see:
- Processed data in `data/processed/`
- Trained models in `models/`
- Evaluation metrics in `reports/metrics.json`
- Visualization plots in `reports/figures/`

### Spam Detection Pipeline

The sample spam dataset (tracked via DVC as `data/raw/spam_train.parquet` and `data/raw/spam_test.parquet`) follows the same four-stage pattern but trains a PyTorch LSTM classifier. Pull the raw Parquet files and reproduce the spam stages just like the IMDB pipeline:

```bash
dvc pull -T data/raw/spam_train.parquet data/raw/spam_test.parquet
dvc repro spam_prepare
dvc repro spam_features
dvc repro spam_train
dvc repro spam_eval
```

Key spam artifacts:
- Cleaned CSVs in `data/processed/spam_*_clean.csv`
- Tensorized datasets in `data/processed/spam_*_features.pt`
- Vocab + weights in `models/spam_vocab.json` and `models/spam_model.pt`
- Metrics + plots in `reports/spam_metrics.json` and `reports/figures/spam_confusion_matrix.png`

### Make Commands

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean compiled files
make clean
```

## Deployment & Docker

### Stack overview
- Single multi-stage image built from `deployment/Dockerfile`; `deployment/docker-compose.yml` reuses it across every service.
- Services: `spam-api` (8000), `spam-model-updater`, `sentiment-api` (8001), `sentiment-model-updater`, and `ui` (8501).
- Shared volumes `spam-models` and `sentiment-models` persist MLflow artifacts so APIs restart quickly.
- Two isolated networks (`spam-net`, `sentiment-net`); the UI is attached to both to reach each API.

<img width="1555" height="850" alt="Captura de pantalla 2025-12-07 152027" src="https://github.com/user-attachments/assets/508c63ec-f984-4882-bb1c-038955dd6682" />

### Environment
1. Copy and fill credentials: `cp example.env .env`
2. Required keys: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT` (defaults to `mlops-imdb`), and `DAGSHUB_ACCESS_KEY`.
3. Useful overrides (defaults shown in the compose file): `SPAM_MODEL_POLL_SECONDS` / `SENTIMENT_MODEL_POLL_SECONDS`, `SPAM_MODEL_DIR` / `SENTIMENT_MODEL_DIR`, `SPAM_MODEL_URI` / `SENTIMENT_MODEL_URI`, and the internal reload URLs (`SPAM_API_RELOAD_URL`, `SENTIMENT_API_RELOAD_URL`).

### Run the full stack
```bash
docker compose -f deployment/docker-compose.yml up --build -d
```
- Spam API: `http://localhost:8000/predict` and `/health`
- Sentiment API: `http://localhost:8001/predict` and `/health`
- UI (Streamlit): `http://localhost:8501`
- Stop everything with `docker compose -f deployment/docker-compose.yml down`; tail logs with `docker compose -f deployment/docker-compose.yml logs -f spam-api` (or any service name).

### Model refresh flow
- Updaters poll MLflow for the latest production-tagged `spam_eval` and `sentiment_eval` runs.
- Artifacts download into a staging directory, then swap atomically into the shared volume (e.g., `/shared/models/spam_model_production`).
- After swapping, the updater calls the API's internal `/reload` on port 9000 so the running server picks up the new model without redeploying.
- Volumes keep the last good model, so API restarts are fast and resilient to temporary MLflow outages.


## Project Organization

```
├── Makefile              <- Common tasks (env creation, lint, test, etc.)
├── README.md             <- Project overview and deployment notes
├── assets/               <- Diagrams and screenshots for docs
├── data/                 <- DVC-managed data (raw/interim/processed/external)
├── deployment/           <- Dockerfile + compose file for the unified stack
├── docker-compose.yml    <- Legacy compose (separate images); prefer deployment/docker-compose.yml
├── docs/                 <- Documentation site content
├── mlops_imdb/           <- Python package (spam + sentiment pipelines, APIs, Streamlit UI)
│   ├── spam/             <- Spam pipeline, API, and updater daemon
│   ├── modeling/         <- Sentiment pipeline, API, and updater daemon
│   └── app.py            <- Streamlit UI that talks to both APIs
├── models/               <- Trained model artifacts (local cache)
├── notebooks/            <- Exploration and experiment notebooks
├── params.yaml           <- Pipeline configuration and hyperparameters
├── reports/              <- Metrics, figures, and evaluation outputs
├── scripts/              <- Helper scripts for data/pipeline tasks
├── tests/                <- Automated tests
├── example.env           <- Template environment variables for MLflow/Dagshub
├── pyproject.toml        <- Project and tooling configuration
├── dvc.yaml              <- DVC pipeline stages and dependencies
├── .pre-commit-config.yaml <- Pre-commit hooks configuration
└── uv.lock               <- Locked dependency graph for reproducible builds
```

