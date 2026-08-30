#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-${ROOT_DIR}/.conda-runtime}"
CONDA_CMD="${CONDA_CMD:-conda}"
CRYOFLEX_OFFLINE="${CRYOFLEX_OFFLINE:-0}"
CRYOFLEX_PYPI_INDEX_URL="${CRYOFLEX_PYPI_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"

if [[ -e "${ENV_PREFIX}" ]]; then
  echo "[create_independent_env] target already exists: ${ENV_PREFIX}" >&2
  echo "Remove it explicitly before requesting a clean rebuild." >&2
  exit 1
fi

offline_args=()
if [[ "${CRYOFLEX_OFFLINE}" == "1" ]]; then
  offline_args+=(--offline)
fi

echo "[1/5] Creating a clean Python 3.8 prefix"
"${CONDA_CMD}" create \
  --yes \
  --prefix "${ENV_PREFIX}" \
  --override-channels \
  --channel defaults \
  "${offline_args[@]}" \
  python=3.8 \
  pip=24.2 \
  cmake \
  make

echo "[2/5] Installing PyTorch 2.1, CUDA 11.8 runtime, and PyTorch3D 0.7.8"
"${CONDA_CMD}" install \
  --yes \
  --prefix "${ENV_PREFIX}" \
  --override-channels \
  --channel pytorch3d \
  --channel nvidia \
  --channel pytorch \
  --channel conda-forge \
  --channel defaults \
  "${offline_args[@]}" \
  pytorch=2.1.0 \
  torchvision=0.16.0 \
  torchaudio=2.1.0 \
  pytorch-cuda=11.8 \
  pytorch3d=0.7.8

echo "[3/5] Installing CryoFlex Python dependencies"
"${ENV_PREFIX}/bin/python" -m pip install \
  --index-url "${CRYOFLEX_PYPI_INDEX_URL}" \
  --disable-pip-version-check \
  --retries 10 \
  --timeout 120 \
  --requirement "${ROOT_DIR}/requirements-core.txt"
"${ENV_PREFIX}/bin/python" -m pip check

echo "[4/5] Checking the bundled SHOT feature binary"
if [[ ! -x "${ROOT_DIR}/alignment/point_cloud_feature" ]]; then
  PATH="${ENV_PREFIX}/bin:${PATH}" bash "${ROOT_DIR}/scripts/build_point_cloud_feature.sh"
else
  echo "[create_independent_env] reusing bundled alignment/point_cloud_feature"
fi
if command -v ldd >/dev/null 2>&1; then
  if ldd "${ROOT_DIR}/alignment/point_cloud_feature" | grep -q "not found"; then
    echo "[create_independent_env] SHOT binary has unresolved shared libraries:" >&2
    ldd "${ROOT_DIR}/alignment/point_cloud_feature" | grep "not found" >&2
    exit 1
  fi
fi

echo "[5/5] Running import and CLI smoke checks"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -c \
  "import easydict, mrcfile, numba, numpy, open3d, pytorch3d, scipy, sklearn, torch, tqdm"
PATH="${ENV_PREFIX}/bin:${PATH}" bash "${ROOT_DIR}/scripts/smoke_test.sh"

cat <<EOF

CryoFlex independent environment is ready.
Logical name: cryoflex
Prefix: ${ENV_PREFIX}

Activate with:
  conda activate ${ENV_PREFIX}

EOF
