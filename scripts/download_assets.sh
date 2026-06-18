#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-models/yolo26n.pt}"
ASSET_BUNDLE_URL="${ASSET_BUNDLE_URL:-}"
GOOGLE_DRIVE_FILE_ID="${GOOGLE_DRIVE_FILE_ID:-}"

mkdir -p "$(dirname "$YOLO_MODEL_PATH")"

if [[ -f "$YOLO_MODEL_PATH" ]]; then
  echo "YOLO model already exists: $YOLO_MODEL_PATH"
  exit 0
fi

if [[ -z "$ASSET_BUNDLE_URL" && -z "$GOOGLE_DRIVE_FILE_ID" ]]; then
  echo "YOLO model is missing: $YOLO_MODEL_PATH" >&2
  echo "Set ASSET_BUNDLE_URL to a direct zip URL, or GOOGLE_DRIVE_FILE_ID to a Google Drive file id." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
BUNDLE="$TMP_DIR/assets.zip"

if [[ -n "$GOOGLE_DRIVE_FILE_ID" ]]; then
  echo "Downloading assets from Google Drive file id: $GOOGLE_DRIVE_FILE_ID"
  URL="https://drive.google.com/uc?export=download&id=${GOOGLE_DRIVE_FILE_ID}"
  curl -L -c "$TMP_DIR/cookies.txt" "$URL" -o "$TMP_DIR/first_response"
  CONFIRM="$(grep -o 'confirm=[0-9A-Za-z_]*' "$TMP_DIR/first_response" | head -n1 | cut -d= -f2 || true)"
  if [[ -n "$CONFIRM" ]]; then
    curl -L -b "$TMP_DIR/cookies.txt" "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${GOOGLE_DRIVE_FILE_ID}" -o "$BUNDLE"
  else
    mv "$TMP_DIR/first_response" "$BUNDLE"
  fi
else
  echo "Downloading assets: $ASSET_BUNDLE_URL"
  curl -L "$ASSET_BUNDLE_URL" -o "$BUNDLE"
fi

if ! unzip -tq "$BUNDLE" >/dev/null; then
  echo "Downloaded file is not a valid zip archive." >&2
  exit 1
fi

unzip -oq "$BUNDLE" -d "$TMP_DIR/unpacked"

FOUND="$(find "$TMP_DIR/unpacked" -type f -name 'yolo26n.pt' | head -n1 || true)"
if [[ -z "$FOUND" ]]; then
  echo "yolo26n.pt not found inside asset bundle." >&2
  echo "Expected zip structure: yolo26n.pt or models/yolo26n.pt" >&2
  exit 1
fi

cp "$FOUND" "$YOLO_MODEL_PATH"
echo "Installed YOLO model: $YOLO_MODEL_PATH"
