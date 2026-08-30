#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-${ROOT_DIR}/work/ntcp_smoke}"
mkdir -p "${WORK_DIR}"

cp /nas_data/donghao/density_to_flex/data/NTCP/7pqg.mrc "${WORK_DIR}/"
cp /nas_data/donghao/density_to_flex/data/NTCP/7pqq.mrc "${WORK_DIR}/"

python "${ROOT_DIR}/module_1.py" \
  --data_dir "${WORK_DIR}" \
  --source 7pqg.mrc \
  --target 7pqq.mrc \
  --source_contour 0.3 \
  --target_contour 0.3 \
  --voxel 1.8

python "${ROOT_DIR}/train.py" \
  --source "${WORK_DIR}/7pqg_1.80_points.npy" \
  --target "${WORK_DIR}/7pqq_1.80_points.npy" \
  --corr "${WORK_DIR}/corr_full_indices_7pqg_7pqq.npy" \
  --output "${WORK_DIR}/pred.npy" \
  --iters 1 \
  --w_sinkhorn 0 \
  --gpu_mode 0

python "${ROOT_DIR}/mid_process.py" \
  --points "${WORK_DIR}/pred.npy" \
  --mrc "${WORK_DIR}/7pqg.mrc" \
  --output "${WORK_DIR}/pred_zyx_voxel.npy"

python "${ROOT_DIR}/mid_process.py" \
  --points "${WORK_DIR}/7pqg_1.80_points.npy" \
  --mrc "${WORK_DIR}/7pqg.mrc" \
  --output "${WORK_DIR}/source_zyx_voxel.npy"

python "${ROOT_DIR}/cpt_flex.py" \
  --source "${WORK_DIR}/source_zyx_voxel.npy" \
  --target "${WORK_DIR}/pred_zyx_voxel.npy" \
  --mrc "${WORK_DIR}/7pqg.mrc" \
  --radius_A 8.0 \
  --mrc_threshold 0.3 \
  --output_npy "${WORK_DIR}/flex.npy" \
  --output_mrc "${WORK_DIR}/flex.mrc" \
  --output_flow_npy "${WORK_DIR}/flow.npy" \
  --output_flow_mrc_prefix "${WORK_DIR}/flow"

echo "Light pipeline completed in ${WORK_DIR}"
