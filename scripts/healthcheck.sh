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
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

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

echo
echo "Checking backend live endpoint"
curl -fsS "http://localhost:${BACKEND_PORT}/api/health/live" >/dev/null

echo "Checking backend readiness"
curl -fsS "http://localhost:${BACKEND_PORT}/api/health/ready"

echo
echo "Checking frontend"
curl -fsS "http://localhost:${FRONTEND_PORT}/" >/dev/null

echo "All services are available. Frontend: http://localhost:${FRONTEND_PORT}"
