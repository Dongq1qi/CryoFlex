#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/alignment/pcl_feature"
BUILD_DIR="${SRC_DIR}/build"
TARGET_BIN="${ROOT_DIR}/alignment/point_cloud_feature"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
cmake --build . -j"$(nproc)"
cp "${BUILD_DIR}/point_cloud_feature" "${TARGET_BIN}"

echo "SHOT feature binary ready: ${TARGET_BIN}"
