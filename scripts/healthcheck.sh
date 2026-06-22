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
FRONTEND_PORT="${FRONTEND_PORT:-18765}"
MODEL_NAME="${MODEL_NAME:-qwen2.5vl:3b}"
OCR_MODE="${OCR_MODE:-qwen}"

compose_args=(-f docker-compose.yml)
if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  compose_args+=(-f docker-compose.gpu.yml)
fi

echo "Docker services:"
docker compose "${compose_args[@]}" ps

echo
echo "Checking Ollama inside Docker network"
docker compose "${compose_args[@]}" exec -T ollama ollama list >/tmp/gradebook_ollama_list.txt
cat /tmp/gradebook_ollama_list.txt
if [[ "$OCR_MODE" == "qwen" ]]; then
  grep -F "$MODEL_NAME" /tmp/gradebook_ollama_list.txt >/dev/null || {
    echo "Required Ollama model is not pulled: $MODEL_NAME" >&2
    exit 1
  }
fi

echo
echo "Checking backend inside Docker network"
docker compose "${compose_args[@]}" exec -T backend curl -fsS http://localhost:8000/api/health/live >/dev/null
docker compose "${compose_args[@]}" exec -T backend curl -fsS http://localhost:8000/api/health/ready

echo
echo "Checking frontend and API proxy"
curl -fsS "http://localhost:${FRONTEND_PORT}/" >/dev/null
curl -fsS "http://localhost:${FRONTEND_PORT}/api/health/live" >/dev/null

echo "All services are available. Frontend: http://localhost:${FRONTEND_PORT}"
