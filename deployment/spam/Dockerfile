FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

# Install dependencies (without project) to maximize cache reuse
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Install project into the already-warmed venv
COPY . /app
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Reuse the prebuilt virtualenv and source tree from the builder
COPY --from=builder /app /app

FROM runtime-base AS spam-api

ENV SPAM_MODEL_URI="models/spam_model_production/spam_model"

EXPOSE 8000

CMD ["sh", "-c", "python3 mlops_imdb/spam/download_production_model.py --output-dir /shared/models/spam_model_production && uvicorn mlops_imdb.spam.api:app --host 0.0.0.0 --port 8000"]

FROM runtime-base AS spam-model-updater

ENV SPAM_MODEL_DIR="/shared/models/spam_model_production" \
    SPAM_MODEL_POLL_SECONDS="300"

CMD ["python3", "mlops_imdb/spam/update_model_daemon.py"]
