#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

DEPLOY_PROFILE="${DEPLOY_PROFILE:-gpu}"
MODEL_NAME="${MODEL_NAME:-qwen2.5vl:3b}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-models/yolo26n.pt}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"

compose_args=(-f docker-compose.yml)
if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  compose_args+=(-f docker-compose.gpu.yml)
fi

mkdir -p data models

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  echo "YOLO model is missing: $YOLO_MODEL_PATH"
  if [[ -n "${ASSET_BUNDLE_URL:-}" || -n "${GOOGLE_DRIVE_FILE_ID:-}" ]]; then
    bash scripts/download_assets.sh
  else
    echo "Put yolo26n.pt into models/ or set ASSET_BUNDLE_URL / GOOGLE_DRIVE_FILE_ID in .env" >&2
    exit 1
  fi
fi

if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  echo "GPU deployment profile selected."
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Install NVIDIA driver and run scripts/install_host_gpu.sh, or use DEPLOY_PROFILE=cpu." >&2
    exit 1
  fi
  nvidia-smi >/dev/null
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is not available. Install Docker first." >&2
  exit 1
fi

echo "[1/4] Starting Ollama"
docker compose "${compose_args[@]}" up -d ollama

echo "[2/4] Waiting for Ollama"
for i in {1..60}; do
  if docker compose "${compose_args[@]}" exec -T ollama ollama list >/dev/null 2>&1; then
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "Ollama did not become ready in time" >&2
    docker compose "${compose_args[@]}" logs --tail=100 ollama >&2
    exit 1
  fi
  sleep 2
done

echo "[3/4] Pulling Ollama model: $MODEL_NAME"
docker compose "${compose_args[@]}" exec -T ollama ollama pull "$MODEL_NAME"

echo "[4/4] Building and starting web services"
docker compose "${compose_args[@]}" up -d --build backend frontend

echo "Waiting for web services"
for i in {1..80}; do
  if bash scripts/healthcheck.sh >/tmp/gradebook_healthcheck.log 2>&1; then
    cat /tmp/gradebook_healthcheck.log
    echo "Deployment completed: http://localhost:${FRONTEND_PORT}"
    exit 0
  fi
  sleep 3
done

cat /tmp/gradebook_healthcheck.log || true
echo "Deployment did not pass health checks" >&2
docker compose "${compose_args[@]}" logs --tail=150 backend frontend ollama >&2
exit 1
