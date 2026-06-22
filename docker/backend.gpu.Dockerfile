# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Heavy dependencies are installed before project sources are copied.
# This keeps Docker layers reusable when only application code changes.
COPY requirements-backend.txt ./requirements-backend.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.7.0" "torchvision>=0.22.0" \
    && python -m pip install -r requirements-backend.txt

COPY packages ./packages
COPY apps ./apps
COPY pyproject.toml* ./

# Project code is installed without dependency resolution because dependencies
# are already cached in the previous layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps -e packages/data_extractor

EXPOSE 8000

CMD ["uvicorn", "apps.web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
