# Multi-stage Dockerfile for nanoGPT-Modern
# Dev stage includes test tools; prod stage includes only runtime deps.

# ---------------------------------------------------------------------------
# Base image
# ---------------------------------------------------------------------------
# CUDA 12.8 aligns with the PyPI default torch 2.8 wheel variant (cu128).
# For other cu12x variants install explicitly, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu126
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Dependencies (shared layer)
# ---------------------------------------------------------------------------
COPY pyproject.toml /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[eval,quant]"

# ---------------------------------------------------------------------------
# Dev stage: includes test/formatting tools
# ---------------------------------------------------------------------------
FROM base AS dev

RUN pip install --no-cache-dir -e ".[dev]"

COPY . /workspace

# Default to running pytest for CI/dev workflows
CMD ["pytest", "-xvs", "--tb=short"]
# ---------------------------------------------------------------------------
# Prod stage: minimal runtime (no dev tools)
# ---------------------------------------------------------------------------

FROM base AS prod

COPY . /workspace

# The api service (docker-compose.yml `api`) serves api_server.py, which needs
# the `server` extra (fastapi/uvicorn/pydantic) — installed here explicitly.
RUN pip install --no-cache-dir -e ".[server]"

# Run pretrain as default (override with docker run args)
CMD ["python", "-m", "training.train_pretrain", "--help"]
