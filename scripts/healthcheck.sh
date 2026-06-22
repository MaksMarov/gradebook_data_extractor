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

FRONTEND_PORT="${FRONTEND_PORT:-18765}"
MODEL_NAME="${MODEL_NAME:-qwen2.5vl:3b}"
OCR_MODE="${OCR_MODE:-qwen}"

printf 'Docker services:\n'
docker compose ps

printf '\nChecking Ollama inside Docker network\n'
docker compose exec -T ollama ollama list >/tmp/gradebook_ollama_list.txt
cat /tmp/gradebook_ollama_list.txt
if [[ "$OCR_MODE" == "qwen" ]]; then
  grep -F "$MODEL_NAME" /tmp/gradebook_ollama_list.txt >/dev/null || {
    echo "Required Ollama model is not pulled: $MODEL_NAME" >&2
    exit 1
  }
fi

printf '\nChecking backend inside Docker network\n'
docker compose exec -T backend curl -fsS http://localhost:8000/api/health/live >/dev/null
docker compose exec -T backend curl -fsS http://localhost:8000/api/health/ready

printf '\nChecking backend GPU status\n'
docker compose exec -T backend curl -fsS http://localhost:8000/api/health/gpu

printf '\nChecking frontend and API proxy\n'
curl -fsS "http://localhost:${FRONTEND_PORT}/" >/dev/null
curl -fsS "http://localhost:${FRONTEND_PORT}/api/health/live" >/dev/null

echo "All services are available. Frontend: http://localhost:${FRONTEND_PORT}"
