#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[create_env] forwarding to the clean, staged installer"
exec bash "${ROOT_DIR}/scripts/create_independent_env.sh" "$@"
