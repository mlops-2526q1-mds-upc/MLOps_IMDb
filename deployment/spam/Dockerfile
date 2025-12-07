FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV="/app/.venv" \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

# Install dependencies (without project) to maximize cache reuse
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Explicitly copy the README and LICENSE required by pyproject.toml
COPY README.md LICENSE /app/
# Copy project into the build context
COPY /mlops_imdb /app/mlops_imdb
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

CMD ["python", "mlops_imdb/spam/api.py"]

FROM runtime-base AS spam-model-updater

ENV SPAM_MODEL_DIR="/shared/models/spam_model_production" \
    SPAM_MODEL_POLL_SECONDS="30"

CMD ["python3", "mlops_imdb/spam/update_model_daemon.py"]
