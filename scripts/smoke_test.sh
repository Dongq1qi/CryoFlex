#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/module_1.py" --help >/dev/null
python "${ROOT_DIR}/train.py" --help >/dev/null
python "${ROOT_DIR}/mid_process.py" --help >/dev/null
python "${ROOT_DIR}/cpt_flex.py" --help >/dev/null

echo "CryoFlex CLI smoke test passed."
