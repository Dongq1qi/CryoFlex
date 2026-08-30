#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-${ROOT_DIR}/.venv}"
PTV3_PREFIX="$(conda env list | awk '$1=="ptv3"{print $NF}')"
PTV3_PYTHON="${PTV3_PREFIX}/bin/python"

if [[ -z "${PTV3_PYTHON}" || ! -x "${PTV3_PYTHON}" ]]; then
  echo "[create_runtime_venv] 无法找到 ptv3 环境中的 python" >&2
  exit 1
fi

"${PTV3_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install numba==0.58.1 easydict==1.13

if [[ ! -x "${ROOT_DIR}/alignment/point_cloud_feature" ]]; then
  echo "[create_runtime_venv] point_cloud_feature 缺失，尝试编译"
  bash "${ROOT_DIR}/scripts/build_point_cloud_feature.sh"
else
  echo "[create_runtime_venv] reusing bundled alignment/point_cloud_feature"
fi

cat <<EOF

CryoFlex runtime venv is ready.
Activate with:
  source ${VENV_DIR}/bin/activate

EOF
