import argparse
import numpy as np
import mrcfile

def get_mrc_voxel_and_origin(mrc_path: str):
    """
    读取 MRC 的 voxel_size(Å/voxel) 和 origin(Å)
    返回:
      voxel_size_A: np.array([vz, vy, vx])  (单位 Å/voxel)
      origin_A:     np.array([oz, oy, ox])  (单位 Å)
    注意：内部统一为 zyx 顺序，方便和你的代码一致。
    """
    with mrcfile.open(mrc_path, permissive=True) as m:
        # voxel size (Å/voxel)
        vs = m.voxel_size  # usually has x,y,z
        vx, vy, vz = float(vs.x), float(vs.y), float(vs.z)

        # origin (Å) - mrcfile 常见提供 header.origin.x/y/z
        # 有些文件 origin 不可靠/为0，这里做个健壮处理
        try:
            ox = float(m.header.origin.x)
            oy = float(m.header.origin.y)
            oz = float(m.header.origin.z)
        except Exception:
            ox = oy = oz = 0.0

    voxel_size_A = np.array([vz, vy, vx], dtype=np.float64)  # zyx
    origin_A     = np.array([oz, oy, ox], dtype=np.float64)  # zyx
    return voxel_size_A, origin_A


def physical_A_to_voxel_zyx(pc_A: np.ndarray, mrc_path: str, input_order: str = "xyz"):
    """
    把物理坐标(Å)点云 -> 体素坐标(voxel index)点云，输出为 (z,y,x)

    参数:
      pc_A: (N,3) 物理坐标点云，单位 Å
      mrc_path: 用于读取 voxel_size 和 origin
      input_order:
          - "xyz": pc_A 每行是 (x,y,z)  (常见)
          - "zyx": pc_A 每行是 (z,y,x)  (如果你自己已经按这个顺序存了)

    返回:
      pc_vox_zyx: (N,3) float64，体素坐标，顺序 (z,y,x)
    """
    if pc_A.ndim != 2 or pc_A.shape[1] != 3:
        raise ValueError("pc_A 必须是 (N,3)")

    voxel_size_A_zyx, origin_A_zyx = get_mrc_voxel_and_origin(mrc_path)

    pc_A = pc_A.astype(np.float64)

    # 统一成 zyx 顺序
    if input_order.lower() == "xyz":
        # (x,y,z) -> (z,y,x)
        pc_A_zyx = pc_A[:, [2, 1, 0]]
    elif input_order.lower() == "zyx":
        pc_A_zyx = pc_A
    else:
        raise ValueError("input_order 只能是 'xyz' 或 'zyx'")

    # voxel index: (coord_A - origin_A) / voxel_size_A
    pc_vox_zyx = (pc_A_zyx - origin_A_zyx[None, :]) / voxel_size_A_zyx[None, :]

    return pc_vox_zyx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="输入 .npy 点云，默认每行为 (x,y,z) 物理坐标")
    parser.add_argument("--mrc", required=True, help="用于读取 voxel size 和 origin 的参考 MRC")
    parser.add_argument("--output", required=True, help="输出体素坐标 .npy，顺序为 (z,y,x)")
    parser.add_argument(
        "--input_order",
        choices=["xyz", "zyx"],
        default="xyz",
        help="输入点云坐标顺序，默认 xyz",
    )
    args = parser.parse_args()

    pc_A = np.load(args.points)
    pc_vox_zyx = physical_A_to_voxel_zyx(pc_A, args.mrc, input_order=args.input_order)
    np.save(args.output, pc_vox_zyx)
    print(f"Saved: {args.output} {pc_vox_zyx.shape}")


if __name__ == "__main__":
    main()
