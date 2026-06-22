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
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-models/yolo26n.pt}"
MODEL_NAME="${MODEL_NAME:-qwen2.5vl:3b}"
FRONTEND_PORT="${FRONTEND_PORT:-18765}"

compose_args=(-f docker-compose.yml)
if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  compose_args+=(-f docker-compose.gpu.yml)
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is not installed"
}

echo "Preflight check"
echo "Profile: $DEPLOY_PROFILE"
echo "Frontend port: $FRONTEND_PORT"
echo "YOLO model: $YOLO_MODEL_PATH"
echo "Ollama model: $MODEL_NAME"

need_cmd docker
need_cmd curl

docker compose version >/dev/null 2>&1 || fail "docker compose plugin is not available"

docker compose "${compose_args[@]}" config >/dev/null

if [[ "$DEPLOY_PROFILE" == "gpu" ]]; then
  need_cmd nvidia-smi
  nvidia-smi >/dev/null || fail "nvidia-smi failed. Check NVIDIA driver on host"
elif [[ "$DEPLOY_PROFILE" != "cpu" ]]; then
  fail "DEPLOY_PROFILE must be gpu or cpu"
fi

mkdir -p data models

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  fail "YOLO model is missing: $YOLO_MODEL_PATH. Copy it to models/yolo26n.pt before deploy or set YOLO_MODEL_PATH in .env."
fi

echo "Preflight OK"
