#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-models/yolo26n.pt}"
OUTPUT="${1:-gradebook-assets.zip}"

if [[ ! -f "$YOLO_MODEL_PATH" ]]; then
  echo "YOLO model not found: $YOLO_MODEL_PATH" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/models"
cp "$YOLO_MODEL_PATH" "$TMP_DIR/models/yolo26n.pt"

(
  cd "$TMP_DIR"
  zip -qr "$ROOT_DIR/$OUTPUT" models/yolo26n.pt
)

echo "Created asset bundle: $OUTPUT"
echo "Upload this zip to Google Drive or a separate model repository and set ASSET_BUNDLE_URL or GOOGLE_DRIVE_FILE_ID in .env."
