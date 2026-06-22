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

YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-models/yolo26n.pt}"
MODEL_NAME="${MODEL_NAME:-qwen2.5vl:3b}"
FRONTEND_PORT="${FRONTEND_PORT:-18765}"
SKIP_GPU_PREFLIGHT="${SKIP_GPU_PREFLIGHT:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is not installed"
}

echo "Preflight check"
echo "Frontend port: $FRONTEND_PORT"
echo "YOLO model: $YOLO_MODEL_PATH"
echo "Ollama model: $MODEL_NAME"

need_cmd docker
need_cmd curl

docker compose version >/dev/null 2>&1 || fail "docker compose plugin is not available"
docker compose config >/dev/null

if [[ "$SKIP_GPU_PREFLIGHT" != "1" ]]; then
  need_cmd nvidia-smi
  nvidia-smi >/dev/null || fail "nvidia-smi failed. Check NVIDIA driver on host"
  echo "Host GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

mkdir -p data models

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  fail "YOLO model is missing: $YOLO_MODEL_PATH. Copy it to models/yolo26n.pt or set YOLO_MODEL_PATH in .env."
fi

echo "Preflight OK"
