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

bash scripts/healthcheck.sh

echo
echo "Checking frontend HTML"
curl -fsS "http://localhost:${FRONTEND_PORT}/" | grep -qi "Gradebook" || {
  echo "Frontend did not return expected HTML" >&2
  exit 1
}

echo "Docker smoke test passed"
