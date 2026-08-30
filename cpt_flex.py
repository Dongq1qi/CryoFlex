#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniform influence (no splat, no weighting):
- 每个浮点点对半径 R(体素) 内的整数网格点施加相同 flex；
- 同一网格点被多个点影响时取均值；
- 仅在 MRC 有效体素 (density > threshold) 上累积；
- 输出 npy，shape 与 MRC 相同，背景与无贡献处填 0。
依赖: numpy, mrcfile
"""

import argparse
import numpy as np
import mrcfile
from typing import Optional, Tuple
from tqdm import tqdm

# ---------- I/O ----------

def load_pc(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} 必须是 (N,3)")
    return arr.astype(np.float64)

def read_mrc_zyx(mrc_path: str):
    with mrcfile.open(mrc_path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32)  # (nz, ny, nx)
        h = m.header
        nx, ny, nz = int(h.nx), int(h.ny), int(h.nz)
        vx = float(h.cella.x) / max(nx, 1)
        vy = float(h.cella.y) / max(ny, 1)
        vz = float(h.cella.z) / max(nz, 1)
    if data.shape != (nz, ny, nx):
        data = np.reshape(data, (nz, ny, nx), order="C")
    return data, (nz, ny, nx), (vx, vy, vz)


def save_like_mrc(reference_mrc: str, output_mrc: str, data_zyx: np.ndarray):
    with mrcfile.open(reference_mrc, permissive=True) as src:
        with mrcfile.new(output_mrc, overwrite=True) as dst:
            dst.set_data(np.asarray(data_zyx, dtype=np.float32))
            dst.voxel_size = src.voxel_size
            dst.header.origin = src.header.origin
            dst.header.nxstart = src.header.nxstart
            dst.header.nystart = src.header.nystart
            dst.header.nzstart = src.header.nzstart
            dst.update_header_from_data()
            dst.update_header_stats()


def save_flow_components_like_mrc(reference_mrc: str, output_prefix: str, flow_zyx: np.ndarray):
    if flow_zyx.ndim != 4 or flow_zyx.shape[-1] != 3:
        raise ValueError("flow_zyx must have shape (nz, ny, nx, 3)")
    component_names = ("z", "y", "x")
    for i, name in enumerate(component_names):
        save_like_mrc(reference_mrc, f"{output_prefix}_{name}.mrc", flow_zyx[..., i])


def resolve_voxel_size_A(
    voxel_size_arg: Optional[float],
    voxel_sizes_from_mrc: Tuple[float, float, float],
    atol: float = 1e-6,
) -> float:
    vx, vy, vz = voxel_sizes_from_mrc

    if min(vx, vy, vz) <= 0:
        raise ValueError(
            "MRC header 中 voxel size 非法，无法自动读取。"
            "请检查 MRC header 或手动传入 --voxel_size_A"
        )

    if not (np.isclose(vx, vy, atol=atol) and np.isclose(vx, vz, atol=atol)):
        raise ValueError(
            f"MRC header 中 voxel size 非各向同性: vx={vx}, vy={vy}, vz={vz}。"
            "当前脚本要求各向同性体素；请先确认数据，或修改脚本支持各向异性。"
        )

    voxel_size_from_mrc = float(vx)

    if voxel_size_arg is None:
        print(f"从 MRC header 读取 voxel_size_A = {voxel_size_from_mrc:.6f} Å/voxel")
        return voxel_size_from_mrc

    voxel_size_arg = float(voxel_size_arg)
    if not np.isclose(voxel_size_arg, voxel_size_from_mrc, atol=atol):
        print(
            "警告: 手动传入的 --voxel_size_A 与 MRC header 不一致，"
            f"将优先使用手动值 {voxel_size_arg:.6f} Å/voxel "
            f"(header: {voxel_size_from_mrc:.6f})"
        )
    return voxel_size_arg



def compute_uniform_influence_grid(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    mrc_zyx: np.ndarray,
    voxel_size_A: float,
    radius_vox: Optional[float] = None,
    radius_A: Optional[float] = None,
    mrc_threshold: float = 0.0,
    use_physical_flex: bool = False,
) -> np.ndarray:
    """
    直接按"半径内均等影响、取均值"的定义把柔性投到体素网格上。
    先计算所有位置的影响情况，最后使用阈值截断保留感兴趣的部分
    输出 (nz,ny,nx)，掩膜外与无贡献处填 0（背景=0）。
    """
    nz, ny, nx = mrc_zyx.shape

    # 半径（体素）
    if radius_vox is None:
        if radius_A is None:
            raise ValueError("需要提供 radius_vox 或 radius_A 其中之一")
        radius_vox = float(radius_A) / float(voxel_size_A)
    R = float(radius_vox)
    if R <= 0:
        raise ValueError("半径必须 > 0")
    R2 = R * R

    # 点级柔性
    disp = target_idx - source_idx
    flex = np.linalg.norm(disp, axis=1)            # voxel
    if use_physical_flex:
        flex = flex * float(voxel_size_A)          # 转 Å

    # 累加器
    grid_sum = np.zeros((nz, ny, nx), dtype=np.float64)
    grid_cnt = np.zeros((nz, ny, nx), dtype=np.uint32)

    # 计算所有位置的影响情况（不进行掩膜过滤）
    print("计算所有位置的影响情况...")
    for i in tqdm(range(source_idx.shape[0]), desc="计算柔性分布", unit="点"):
        pz, py, px = source_idx[i]  # source_idx 存储的是 (z,y,x) 顺序
        v = float(flex[i])

        # 搜索盒（整数索引）- 不进行掩膜过滤
        x0 = max(0, int(np.ceil (px - R)))
        x1 = min(nx-1, int(np.floor(px + R)))
        y0 = max(0, int(np.ceil (py - R)))
        y1 = min(ny-1, int(np.floor(py + R)))
        z0 = max(0, int(np.ceil (pz - R)))
        z1 = min(nz-1, int(np.floor(pz + R)))

        for zz in range(z0, z1+1):
            dz2 = (zz - pz) * (zz - pz)
            for yy in range(y0, y1+1):
                dy2 = (yy - py) * (yy - py)
                for xx in range(x0, x1+1):
                    dx2 = (xx - px) * (xx - px)
                    if dx2 + dy2 + dz2 <= R2:
                        grid_sum[zz, yy, xx] += v
                        grid_cnt[zz, yy, xx] += 1

    # 取均值
    print("归一化计算...")
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    valid = grid_cnt > 0
    out[valid] = (grid_sum[valid] / grid_cnt[valid]).astype(np.float32)

    # 最后使用阈值截断：只保留MRC掩膜内的区域
    print(f"应用MRC阈值截断 (threshold={mrc_threshold})...")
    mask = mrc_zyx > float(mrc_threshold)
    out = out * mask.astype(np.float32)  # 掩膜外设为0

    return out


def compute_weighted_influence_grid(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    mrc_zyx: np.ndarray,
    voxel_size_A: float,
    radius_vox: Optional[float] = None,
    radius_A: Optional[float] = None,
    mrc_threshold: float = 0.0,
    use_physical_flex: bool = False,
    weighting_scheme: str = "logistic",
    weighting_params: Optional[dict] = None,
) -> np.ndarray:
    """
    加权柔性分布计算：距离越近的影响越大
    先计算所有位置的影响情况，最后使用阈值截断保留感兴趣的部分
    输出 (nz,ny,nx)，背景与无贡献处填 0。

    weighting_scheme:
    - "gaussian": 高斯权重，权重随距离指数衰减
    - "inverse": 距离倒数权重，权重按1/distance衰减
    - "linear": 线性衰减权重，权重随距离线性减少
    - "logistic": Logistic/Sigmoid权重，S形衰减曲线（默认）
    - "uniform": 均匀权重（等价于原算法）

    对于 "logistic" 方案，可通过 weighting_params 设置参数：
    - "d0": 拐点位置（默认 0.6R）
    - "k": 陡峭度参数（默认 0.08R，越小越陡峭）
    """
    nz, ny, nx = mrc_zyx.shape

    # 半径（体素）
    if radius_vox is None:
        if radius_A is None:
            raise ValueError("需要提供 radius_vox 或 radius_A 其中之一")
        radius_vox = float(radius_A) / float(voxel_size_A)
    R = float(radius_vox)
    if R <= 0:
        raise ValueError("半径必须 > 0")
    R2 = R * R

    # 点级柔性
    disp = target_idx - source_idx
    flex = np.linalg.norm(disp, axis=1)
    if use_physical_flex:
        flex = flex * float(voxel_size_A)

    # 设置权重参数
    if weighting_params is None:
        weighting_params = {}
    sigma = weighting_params.get('sigma', R / 3.0)  # 高斯标准差（默认 R/3）
    min_weight = weighting_params.get('min_weight', 1e-6)  # 最小权重阈值（全局默认 1e-6）
    cap_weight_max = weighting_params.get('cap_weight_max', 1.0)  # 最大权重上限（默认 1.0）

    # 累加器 - 使用权重累加
    grid_weighted_sum = np.zeros((nz, ny, nx), dtype=np.float64)  # 加权柔性总和
    grid_weight_sum = np.zeros((nz, ny, nx), dtype=np.float64)    # 权重总和

    # 计算所有位置的影响情况（不进行掩膜过滤）
    print("计算所有位置的影响情况...")
    for i in tqdm(range(source_idx.shape[0]), desc="计算加权柔性分布", unit="点"):
        pz, py, px = source_idx[i]
        v = float(flex[i])

        # 搜索盒（整数索引）- 不进行掩膜过滤
        x0 = max(0, int(np.ceil(px - R)))
        x1 = min(nx-1, int(np.floor(px + R)))
        y0 = max(0, int(np.ceil(py - R)))
        y1 = min(ny-1, int(np.floor(py + R)))
        z0 = max(0, int(np.ceil(pz - R)))
        z1 = min(nz-1, int(np.floor(pz + R)))

        for zz in range(z0, z1+1):
            dz2 = (zz - pz) * (zz - pz)
            for yy in range(y0, y1+1):
                dy2 = (yy - py) * (yy - py)
                for xx in range(x0, x1+1):
                    dx2 = (xx - px) * (xx - px)
                    distance_squared = dx2 + dy2 + dz2
                    if distance_squared <= R2:
                        distance = np.sqrt(distance_squared)

                        # 计算权重
                        if weighting_scheme == "gaussian":
                            weight = np.exp(-distance_squared / (2 * sigma**2))
                        elif weighting_scheme == "inverse":
                            # 避免 d→0 处无界，后续再统一裁剪到 cap_weight_max
                            weight = 1.0 / (distance + 1e-8)
                        elif weighting_scheme == "linear":
                            weight = 1.0 - (distance / R)
                        elif weighting_scheme == "logistic":
                            # Logistic/Sigmoid 权重: w(d) = 1 / (1 + exp((d - d0) / k))
                            d0 = weighting_params.get('d0', 0.6 * R)   # 拐点位置，默认0.6倍半径
                            k = weighting_params.get('k', 0.08 * R)    # 陡峭度，默认0.08倍半径
                            weight = 1.0 / (1.0 + np.exp((distance - d0) / k))
                        elif weighting_scheme == "uniform":
                            weight = 1.0
                        else:
                            raise ValueError(f"未知的权重方案: {weighting_scheme}")

                        # 幅度约束：先上限裁剪再下限过滤
                        if cap_weight_max is not None:
                            weight = min(weight, float(cap_weight_max))
                        # 只累加有意义的权重
                        if weight > min_weight:
                            grid_weighted_sum[zz, yy, xx] += v * weight
                            grid_weight_sum[zz, yy, xx] += weight

    # 归一化：加权平均
    print("归一化计算...")
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    valid = grid_weight_sum > 0
    out[valid] = (grid_weighted_sum[valid] / grid_weight_sum[valid]).astype(np.float32)

    # 最后使用阈值截断：只保留MRC掩膜内的区域
    print(f"应用MRC阈值截断 (threshold={mrc_threshold})...")
    mask = mrc_zyx > float(mrc_threshold)
    out = out * mask.astype(np.float32)  # 掩膜外设为0

    return out


def compute_weighted_flow_and_flex_grids(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    mrc_zyx: np.ndarray,
    voxel_size_A: float,
    radius_vox: Optional[float] = None,
    radius_A: Optional[float] = None,
    mrc_threshold: float = 0.0,
    use_physical_flex: bool = False,
    weighting_scheme: str = "logistic",
    weighting_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将点级位移向量和位移幅度同时扩散回体素网格。

    返回:
    - flex_grid: (nz, ny, nx)，与旧实现一致，为点级位移幅度的加权平均
    - flow_grid: (nz, ny, nx, 3)，分量顺序为 (dz, dy, dx)

    当 use_physical_flex=True 时，flex 和 flow 都使用 Å；否则使用 voxel。
    """
    nz, ny, nx = mrc_zyx.shape

    if radius_vox is None:
        if radius_A is None:
            raise ValueError("需要提供 radius_vox 或 radius_A 其中之一")
        radius_vox = float(radius_A) / float(voxel_size_A)
    R = float(radius_vox)
    if R <= 0:
        raise ValueError("半径必须 > 0")
    R2 = R * R

    disp = target_idx - source_idx
    flex = np.linalg.norm(disp, axis=1)
    if use_physical_flex:
        disp = disp * float(voxel_size_A)
        flex = flex * float(voxel_size_A)

    if weighting_params is None:
        weighting_params = {}
    sigma = weighting_params.get("sigma", R / 3.0)
    min_weight = weighting_params.get("min_weight", 1e-6)
    cap_weight_max = weighting_params.get("cap_weight_max", 1.0)

    grid_flex_sum = np.zeros((nz, ny, nx), dtype=np.float64)
    grid_flow_sum = np.zeros((nz, ny, nx, 3), dtype=np.float64)
    grid_weight_sum = np.zeros((nz, ny, nx), dtype=np.float64)

    print("计算所有位置的 flow/flex 影响情况...")
    for i in tqdm(range(source_idx.shape[0]), desc="计算 flow/flex 分布", unit="点"):
        pz, py, px = source_idx[i]
        flow_i = disp[i]
        flex_i = float(flex[i])

        x0 = max(0, int(np.ceil(px - R)))
        x1 = min(nx - 1, int(np.floor(px + R)))
        y0 = max(0, int(np.ceil(py - R)))
        y1 = min(ny - 1, int(np.floor(py + R)))
        z0 = max(0, int(np.ceil(pz - R)))
        z1 = min(nz - 1, int(np.floor(pz + R)))

        for zz in range(z0, z1 + 1):
            dz2 = (zz - pz) * (zz - pz)
            for yy in range(y0, y1 + 1):
                dy2 = (yy - py) * (yy - py)
                for xx in range(x0, x1 + 1):
                    dx2 = (xx - px) * (xx - px)
                    distance_squared = dx2 + dy2 + dz2
                    if distance_squared <= R2:
                        distance = np.sqrt(distance_squared)

                        if weighting_scheme == "gaussian":
                            weight = np.exp(-distance_squared / (2 * sigma**2))
                        elif weighting_scheme == "inverse":
                            weight = 1.0 / (distance + 1e-8)
                        elif weighting_scheme == "linear":
                            weight = 1.0 - (distance / R)
                        elif weighting_scheme == "logistic":
                            d0 = weighting_params.get("d0", 0.6 * R)
                            k = weighting_params.get("k", 0.08 * R)
                            weight = 1.0 / (1.0 + np.exp((distance - d0) / k))
                        elif weighting_scheme == "uniform":
                            weight = 1.0
                        else:
                            raise ValueError(f"未知的权重方案: {weighting_scheme}")

                        if cap_weight_max is not None:
                            weight = min(weight, float(cap_weight_max))
                        if weight > min_weight:
                            grid_flex_sum[zz, yy, xx] += flex_i * weight
                            grid_flow_sum[zz, yy, xx, :] += flow_i * weight
                            grid_weight_sum[zz, yy, xx] += weight

    print("归一化计算 flow/flex...")
    flex_grid = np.zeros((nz, ny, nx), dtype=np.float32)
    flow_grid = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    valid = grid_weight_sum > 0
    flex_grid[valid] = (grid_flex_sum[valid] / grid_weight_sum[valid]).astype(np.float32)
    flow_grid[valid] = (grid_flow_sum[valid] / grid_weight_sum[valid, None]).astype(np.float32)

    print(f"应用MRC阈值截断 (threshold={mrc_threshold})...")
    mask = mrc_zyx > float(mrc_threshold)
    flex_grid = flex_grid * mask.astype(np.float32)
    flow_grid = flow_grid * mask[..., None].astype(np.float32)

    return flex_grid, flow_grid


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source.npy (N,3) 索引坐标/voxel, 默认要求是顺序是zyx的排列顺序")
    ap.add_argument("--target", required=True, help="target.npy (N,3) 索引坐标/voxel")
    ap.add_argument("--mrc", required=True, help="体素对齐的 MRC 文件（提供掩膜与尺寸）")
    ap.add_argument(
        "--voxel_size_A",
        type=float,
        default=None,
        help="Å/voxel；默认从 MRC header 自动读取，手动传入时优先使用该值",
    )

    # 半径二选一：体素 或 Å（都给则优先用体素）
    ap.add_argument("--radius_vox", type=float, default=None, help="影响半径（体素）")
    ap.add_argument("--radius_A", type=float, default=None, help="影响半径（Å）")

    ap.add_argument("--mrc_threshold", type=float, default=0.0, help="<= 阈值视为背景")
    ap.add_argument("--use_physical_flex", action="store_true", help="柔性用 Å（默认用 voxel）")
    ap.add_argument("--output_npy", required=True, help="输出 .npy 路径")
    ap.add_argument("--output_mrc", default=None, help="可选输出 .mrc 路径")
    ap.add_argument(
        "--output_flow_npy",
        default=None,
        help="可选输出 flow .npy 路径，shape=(nz,ny,nx,3)，分量顺序为(dz,dy,dx)",
    )
    ap.add_argument(
        "--output_flow_mrc_prefix",
        default=None,
        help="可选输出 flow 分量 MRC 前缀，会写出 <prefix>_z.mrc/_y.mrc/_x.mrc",
    )

    # 加权算法选择与可选参数
    ap.add_argument(
        "--weighting_scheme",
        type=str,
        choices=["gaussian", "inverse", "linear", "logistic", "uniform"],
        default="gaussian",
        help="加权方案：gaussian/inverse/linear/logistic/uniform",
    )
    ap.add_argument("--sigma", type=float, default=None, help="gaussian 标准差（体素）")
    ap.add_argument("--min_weight", type=float, default=None, help="最小权重阈值，默认 1e-6")
    ap.add_argument("--d0", type=float, default=None, help="logistic 拐点位置 d0（体素，默认 0.6R）")
    ap.add_argument("--k", type=float, default=None, help="logistic 陡峭度 k（体素，默认 0.08R）")
    args = ap.parse_args()

    source_idx = load_pc(args.source)
    target_idx = load_pc(args.target)
    if source_idx.shape != target_idx.shape:
        raise ValueError("source 与 target 形状不一致")

    mrc_zyx, (nz, ny, nx), (vx, vy, vz) = read_mrc_zyx(args.mrc)
    voxel_size_A = resolve_voxel_size_A(args.voxel_size_A, (vx, vy, vz))

    # 组装 weighting_params
    weighting_params = None
    if any(v is not None for v in [args.sigma, args.min_weight, args.d0, args.k]):
        weighting_params = {}
        if args.sigma is not None:
            weighting_params["sigma"] = float(args.sigma)
        if args.min_weight is not None:
            weighting_params["min_weight"] = float(args.min_weight)
        if args.d0 is not None:
            weighting_params["d0"] = float(args.d0)
        if args.k is not None:
            weighting_params["k"] = float(args.k)

    should_output_flow = args.output_flow_npy is not None or args.output_flow_mrc_prefix is not None
    if should_output_flow:
        grid, flow_grid = compute_weighted_flow_and_flex_grids(
            source_idx=source_idx,
            target_idx=target_idx,
            mrc_zyx=mrc_zyx,
            voxel_size_A=voxel_size_A,
            radius_vox=args.radius_vox,
            radius_A=args.radius_A,
            mrc_threshold=args.mrc_threshold,
            use_physical_flex=args.use_physical_flex,
            weighting_scheme=args.weighting_scheme,
            weighting_params=weighting_params,  # None 则使用默认参数
        )
    else:
        grid = compute_weighted_influence_grid(
            source_idx=source_idx,
            target_idx=target_idx,
            mrc_zyx=mrc_zyx,
            voxel_size_A=voxel_size_A,
            radius_vox=args.radius_vox,
            radius_A=args.radius_A,
            mrc_threshold=args.mrc_threshold,
            use_physical_flex=args.use_physical_flex,
            weighting_scheme=args.weighting_scheme,
            weighting_params=weighting_params,  # None 则使用默认参数
        )
        flow_grid = None

    # 保存为 .npy（shape 与 mrc 相同，背景=0）
    np.save(args.output_npy, grid)
    print(f"[OK] 保存 .npy: {args.output_npy}")
    if args.output_mrc:
        save_like_mrc(args.mrc, args.output_mrc, grid)
        print(f"[OK] 保存 .mrc: {args.output_mrc}")
    if args.output_flow_npy:
        np.save(args.output_flow_npy, flow_grid)
        print(f"[OK] 保存 flow .npy: {args.output_flow_npy} (shape={flow_grid.shape}, components=dz,dy,dx)")
    if args.output_flow_mrc_prefix:
        save_flow_components_like_mrc(args.mrc, args.output_flow_mrc_prefix, flow_grid)
        print(f"[OK] 保存 flow MRC 分量: {args.output_flow_mrc_prefix}_z.mrc/_y.mrc/_x.mrc")
    print(
        f"  shape={grid.shape} (nz,ny,nx) 与 MRC 一致，背景=0,"
        f" 单位={'Å' if args.use_physical_flex else 'voxel'},"
        f" 加权方案={args.weighting_scheme}"
    )



if __name__ == "__main__":
    main()
