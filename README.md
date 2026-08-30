# CryoFlex

`CryoFlex` 是从 `/nas_data/donghao/density_to_flex` 主流程中抽出的独立运行版本，只保留
`module_1.py -> train.py -> mid_process.py -> cpt_flex.py` 这条主链路所需的代码和资源。

## 主流程

1. `module_1.py`
   - 从 source/target MRC 采样点云
   - 进行 meanshift + DBSCAN 提取关键点
   - 计算 SHOT 特征并建立关键点对应
2. `train.py`
   - 用层级形变网络做点云配准
   - 优化 Chamfer、Sinkhorn、对应点约束、局部运动一致性
3. `mid_process.py`
   - 把 `train.py` 输出的物理坐标 `(x,y,z)` 转换成 `cpt_flex.py` 需要的体素坐标 `(z,y,x)`
4. `cpt_flex.py`
   - 把源/目标点对的位移投回 MRC 体素网格
   - 生成 flexibility map，并可同时输出 voxel-level flow map

## 环境策略

这个独立版本的环境组合参考了两条线：

- `CryoAlign` 官方安装栈：保留 `open3d / scipy / scikit-learn / mrcfile` 这类预处理依赖
- `ptv3` 环境：采用更稳定的新 `torch + pytorch3d` 组合，保证训练阶段能单独运行

这里不再依赖 `teaserpp-python`；主流程实际不需要它。

## 快速开始

推荐从空目录创建项目内独立 Conda 环境：

```bash
cd /nas_data/donghao/cryoflex
bash scripts/create_independent_env.sh
conda activate /nas_data/donghao/cryoflex/.conda-runtime
```

这条路径会：

- 从 Python 3.8 基础环境开始逐层安装依赖，不使用 `conda clone`
- 安装独立的 PyTorch 2.1、CUDA 11.8 runtime、PyTorch3D 0.7.8 和 Open3D 0.19
- 默认使用中科大 PyPI 镜像加速 Python 包下载
- 不继承 `.venv`、`ptv3` 或 `cryoalign` 的 `site-packages`

环境的逻辑名称是 `cryoflex`，实际 prefix 是项目内的 `.conda-runtime`。
详细安装、镜像切换和验证方法见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)。

## 示例命令

### 1. 点云采样与关键点对应

```bash
python module_1.py \
  --data_dir /path/to/data \
  --source source.mrc \
  --target target.mrc \
  --source_contour 0.267 \
  --target_contour 0.267 \
  --voxel 3.0
```

### 2. 点云配准训练

```bash
python train.py \
  --source /path/to/data/source_3.00_points.npy \
  --target /path/to/data/target_3.00_points.npy \
  --corr /path/to/data/corr_full_indices_source_target.npy \
  --output /path/to/data/pred.npy \
  --w_sinkhorn 1000 \
  --gpu_mode 1
```

当前训练默认使用 `w_sinkhorn=1000`；使用 CPU 时传入 `--gpu_mode 0`。
`module_1.py` 同时保存关键点内部索引 `corr_indices_*` 和可供训练使用的
完整采样点索引 `corr_full_indices_*`；`train.py --corr` 必须使用后者。

### 3. 物理坐标转体素坐标

```bash
python mid_process.py \
  --points /path/to/data/pred.npy \
  --mrc /path/to/data/source.mrc \
  --output /path/to/data/pred_zyx_voxel.npy
```

### 4. 计算 flexibility map

```bash
python cpt_flex.py \
  --source /path/to/data/source_zyx_voxel.npy \
  --target /path/to/data/pred_zyx_voxel.npy \
  --mrc /path/to/data/source.mrc \
  --radius_A 10.0 \
  --mrc_threshold 0.0 \
  --output_npy /path/to/data/flex.npy \
  --output_mrc /path/to/data/flex.mrc \
  --output_flow_npy /path/to/data/flow.npy \
  --output_flow_mrc_prefix /path/to/data/flow
```

`flow.npy` 的 shape 为 `(nz, ny, nx, 3)`，分量顺序为 `(dz, dy, dx)`。
`--output_flow_mrc_prefix /path/to/data/flow` 会额外写出
`flow_z.mrc`、`flow_y.mrc` 和 `flow_x.mrc`。

### 5. 在 Cα 位置采样 voxel-level flow

```bash
python scripts/sample_ca_flow.py \
  --flow_npy /path/to/data/flow.npy \
  --reference_mrc /path/to/data/source.mrc \
  --pdb /path/to/model_aligned_to_source.pdb \
  --output_csv /path/to/data/ca_flow.csv \
  --method trilinear
```

输出 CSV 包含 Cα 的坐标、fractional voxel index、`flow_dz/dy/dx`，
以及便于和原子模型 displacement 比较的 `flow_x/y/z` 和 `flow_mag`。
也可以使用 `--flow_mrc_prefix /path/to/data/flow` 读取
`flow_z.mrc`、`flow_y.mrc`、`flow_x.mrc`。

## 重新编译 SHOT 特征程序

```bash
bash scripts/build_point_cloud_feature.sh
```
/bin/bash: -c: line 2: unexpected EOF while looking for matching `''
/bin/bash: -c: line 3: syntax error: unexpected end of file
