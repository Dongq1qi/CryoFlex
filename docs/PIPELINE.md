# Pipeline Notes

## 输入输出约定

- `module_1.py`
  - 输入：成对 `MRC`
  - 输出：
    - `*_points.npy`
    - `*_normals.npy`
    - `*_keypoints.npy`
    - `corr_indices_*.npy`：SHOT 关键点数组内部索引，仅用于追踪匹配来源
    - `corr_full_indices_*.npy`：映射到完整采样点云的索引，供 `train.py --corr` 使用
    - `corr_full_mapping_distances_*.npy`：关键点到完整采样点的最近邻距离（Å）
- `train.py`
  - 输入：source/target 采样点云，及可选对应关系
  - `--corr` 必须传入完整采样点索引 `corr_full_indices_*.npy`，不能直接传入关键点内部索引
  - 输出：配准后的预测点云 `pred.npy`
  - 当前损失：Chamfer + Sinkhorn + KNN motion consistency + 可选 correspondence anchor
  - 当前默认：`w_cd=1`、`w_sinkhorn=1000`、`w_knn_motion=1`、`w_ldmk=1`
  - 可覆盖参数：`--seed`、`--iters`、`--w_sinkhorn`、`--w_knn_motion`、`--w_ldmk`、`--corr_loss_levels`、`--gpu_mode`
- `mid_process.py`
  - 输入：物理坐标点云 `(x,y,z)` 与参考 `MRC`
  - 输出：体素坐标点云 `(z,y,x)`
- `cpt_flex.py`
  - 输入：source/target 体素坐标点云，参考 `MRC`
  - 输出：`flex.npy`，可选 `flex.mrc`
  - 可选输出：`flow.npy`，shape 为 `(nz, ny, nx, 3)`，分量顺序为 `(dz, dy, dx)`；也可写出 `flow_z.mrc`、`flow_y.mrc`、`flow_x.mrc`
  - 支持 `gaussian`、`inverse`、`linear`、`logistic`、`uniform` 五种扩散权重
  - 默认输出单位为 voxel；传入 `--use_physical_flex` 后，flex 和 flow 使用 Å
  - `--radius_vox` 与 `--radius_A` 二选一；同时提供时优先使用 `--radius_vox`
- `scripts/sample_ca_flow.py`
  - 输入：`flow.npy` 或 `flow_z/y/x.mrc`，参考 `MRC`，以及与该 map 对齐的 PDB
  - 输出：Cα-level flow CSV，包含 `flow_dz/dy/dx`、`flow_x/y/z` 和 `flow_mag`

## 坐标规则

- `module_1.py` 和 `train.py` 中的点云默认是物理坐标 `(x, y, z)`，单位为 Å
- `cpt_flex.py` 需要的是体素坐标 `(z, y, x)`
- 因此主流程里 `mid_process.py` 是必需步骤

## 与原仓库的关系

- 这个目录不依赖 `/nas_data/donghao/density_to_flex` 的相对路径
- 只要环境装好、二进制可用，整个流程可以从本目录独立运行
- 逐阶段同步结论见 `docs/PIPELINE_SYNC_AUDIT_20260726.md`
