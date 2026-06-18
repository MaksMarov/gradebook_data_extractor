FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY packages ./packages
COPY apps ./apps
COPY pyproject.toml* ./
COPY requirements*.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
    && pip install -e packages/data_extractor \
    && pip install fastapi uvicorn[standard] python-multipart

EXPOSE 8000

CMD ["uvicorn", "apps.web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
