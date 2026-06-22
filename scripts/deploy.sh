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
FRONTEND_PORT="${FRONTEND_PORT:-18765}"

compose_args=(-f docker-compose.yml)
if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  compose_args+=(-f docker-compose.gpu.yml)
fi

mkdir -p data models

bash scripts/preflight.sh

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  echo "YOLO model is missing: $YOLO_MODEL_PATH" >&2
  echo "Put the model file into models/yolo26n.pt or set YOLO_MODEL_PATH in .env." >&2
  exit 1
fi

echo "[1/5] Starting Ollama"
docker compose "${compose_args[@]}" up -d ollama

echo "[2/5] Waiting for Ollama"
for i in {1..90}; do
  if docker compose "${compose_args[@]}" exec -T ollama ollama list >/dev/null 2>&1; then
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    echo "Ollama did not become ready in time" >&2
    docker compose "${compose_args[@]}" logs --tail=120 ollama >&2
    exit 1
  fi
  sleep 2
done

echo "[3/5] Pulling Ollama model: $MODEL_NAME"
docker compose "${compose_args[@]}" exec -T ollama ollama pull "$MODEL_NAME"

echo "[4/5] Building and starting web services"
docker compose "${compose_args[@]}" up -d --build backend frontend

echo "[5/5] Waiting for health checks"
for i in {1..100}; do
  if bash scripts/healthcheck.sh >/tmp/gradebook_healthcheck.log 2>&1; then
    cat /tmp/gradebook_healthcheck.log
    echo "Deployment completed: http://localhost:${FRONTEND_PORT}"
    exit 0
  fi
  sleep 3
done

cat /tmp/gradebook_healthcheck.log || true
echo "Deployment did not pass health checks" >&2
docker compose "${compose_args[@]}" logs --tail=180 backend frontend ollama >&2
exit 1
