# CryoFlex 独立环境安装

## 环境名称与位置

- 逻辑名称：`cryoflex`
- 项目内 prefix：`/nas_data/donghao/cryoflex/.conda-runtime`
- Python：3.8
- PyTorch/CUDA：PyTorch 2.1.0 + CUDA 11.8 runtime
- PyTorch3D：0.7.8
- Open3D：0.19.0

该环境从空目录创建，不使用 `conda clone`，也不启用
`--system-site-packages`。它不会从 `.venv`、`ptv3` 或 `cryoalign`
导入 Python 包。

## 从 0 到 1 安装

```bash
cd /nas_data/donghao/cryoflex
bash scripts/create_independent_env.sh
conda activate /nas_data/donghao/cryoflex/.conda-runtime
```

脚本按以下顺序执行：

1. 创建全新的 Python 3.8 Conda prefix。
2. 安装 PyTorch、CUDA runtime、torchvision、torchaudio 和 PyTorch3D。
3. 从 `requirements-core.txt` 安装 CryoFlex Python 依赖。
4. 检查项目自带的 SHOT 特征程序及其动态库。
5. 执行核心模块导入和四段 pipeline CLI smoke test。

默认使用中科大 PyPI 镜像。需要切换到其他兼容的 PyPI 源时：

```bash
CRYOFLEX_PYPI_INDEX_URL=https://pypi.org/simple \
  bash scripts/create_independent_env.sh
```

如果目标机器已经完整缓存 Conda 包，但外网 channel 不稳定，可以使用：

```bash
CRYOFLEX_OFFLINE=1 bash scripts/create_independent_env.sh
```

`CRYOFLEX_OFFLINE=1` 只影响 Conda 阶段；pip 仍从配置的 PyPI 镜像安装。

## 目标目录已存在时

安装脚本会拒绝覆盖已有 prefix。若确实要重建，请先自行备份并明确删除
`.conda-runtime`，然后重新运行脚本。脚本不会自动删除环境，也不会把旧环境
clone 到新环境。

## 安装后验证

```bash
cd /nas_data/donghao/cryoflex
conda activate /nas_data/donghao/cryoflex/.conda-runtime

python -m pip check
bash scripts/smoke_test.sh
python -m unittest tests.test_cpt_flex_flow tests.test_sample_ca_flow -v
```

确认解释器和模块来源：

```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import os
import sys
import open3d
import pytorch3d
import torch

print(sys.executable)
print(sys.prefix)
for module in (torch, pytorch3d, open3d):
    print(os.path.realpath(module.__file__))
PY
```

所有路径都应位于 `/nas_data/donghao/cryoflex/.conda-runtime`。

## 可复现清单

- `environment.yml`：声明式环境参考
- `requirements-core.txt`：CryoFlex Python 直接依赖及固定版本
- `environment-lock-linux-64.txt`：本次通过验证的 Conda explicit lock
- `requirements-lock-pip.txt`：本次通过验证的完整 pip freeze
- `docs/INDEPENDENT_ENV_AUDIT_20260726.md`：安装和测试证据

Conda explicit lock 与操作系统平台相关；在不同平台上优先运行
`scripts/create_independent_env.sh`，不要直接复用 Linux lock。
