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

DAYS="${CLEANUP_JOBS_DAYS:-14}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)
      DAYS="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[[ "$DAYS" =~ ^[0-9]+$ ]] || { echo "--days must be a non-negative integer" >&2; exit 1; }

JOBS_DIR="${WEB_JOBS_DIR:-data/web_jobs}"
mkdir -p "$JOBS_DIR"

case "$(realpath -m "$JOBS_DIR")" in
  "$ROOT_DIR"|"/"|"$ROOT_DIR/data")
    echo "Refusing to cleanup unsafe directory: $JOBS_DIR" >&2
    exit 1
    ;;
esac

echo "Cleaning jobs older than $DAYS day(s) in: $JOBS_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$DAYS" -print
  exit 0
fi

find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +"$DAYS" -print -exec rm -rf {} +
echo "Cleanup complete"
