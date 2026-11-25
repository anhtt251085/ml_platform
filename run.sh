#!/usr/bin/env bash
# Start the SAI Platform (API + UI) on :8000.
set -euo pipefail
cd "$(dirname "$0")"
export SAI_PLATFORM_DATA="${SAI_PLATFORM_DATA:-$PWD/var}"
exec uvicorn backend.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
