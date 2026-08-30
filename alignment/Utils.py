from pathlib import Path
import re
import subprocess

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_xyz(file_path):
    return o3d.io.read_point_cloud(file_path, format="xyz")


def load_sample(file_path, density=False):
    point_list = []
    vector_list = []
    density_list = []
    with open(file_path, "r") as handle:
        lines = handle.readlines()
    sample = float(lines[0].strip())
    origin_x, origin_y, origin_z = [float(v) for v in lines[3].strip().split()]
    for i in range(5, len(lines)):
        line = lines[i]
        if i % 2:
            _, x, y, z = line.strip().split()
            point_list.append(
                [
                    float(x) * sample + origin_x,
                    float(y) * sample + origin_y,
                    float(z) * sample + origin_z,
                ]
            )
        else:
            v_x, v_y, v_z, d = line.strip().split()
            vector_list.append([float(v_x), float(v_y), float(v_z)])
            density_list.append([float(d)])
    if density:
        return np.array(point_list), np.array(vector_list), np.array(density_list)
    return np.array(point_list), np.array(vector_list)


def find_nn_cpu(feat0, feat1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1)
    if return_distance:
        return nn_inds, dists
    return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_nn_cpu(feats0, feats1)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_nn_cpu(feats1, feats0)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = corres10_idx0[corres01_idx1] == corres01_idx0
    return corres01_idx0[mutual_filter], corres01_idx1[mutual_filter]


def txt2pcd(pcd_points, output):
    with open(output, "w") as handle:
        handle.writelines("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.writelines("VERSION 0.7\n")
        handle.writelines("FIELDS x y z\n")
        handle.writelines("SIZE 4 4 4\n")
        handle.writelines("TYPE F F F\n")
        handle.writelines("COUNT 1 1 1\n")
        handle.writelines("WIDTH " + str(pcd_points.shape[0]) + "\n")
        handle.writelines("HEIGHT 1\n")
        handle.writelines("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.writelines("POINTS " + str(pcd_points.shape[0]) + "\n")
        handle.writelines("DATA ascii\n")
        for point in pcd_points:
            handle.write(" ".join([f"{value:.5f}" for value in point]))
            handle.write("\n")


def read_features(feature_dir, mode="SHOT", key=False):
    features = []
    with open(feature_dir, "r") as handle:
        lines = handle.readlines()
    for line in lines:
        if not key:
            if mode in {"SHOT", "3DSC", "USC"}:
                values = line.strip().split("(")[2][:-1]
            else:
                values = line.strip()[1:-1]
        else:
            if mode == "SHOT":
                values = line.strip().split("(")[2][:-1]
            else:
                values = line.strip()[1:-1]
        features.append([float(v) for v in values.split(",")])
    return np.stack(features, axis=0)


def _feature_binary_path():
    return Path(__file__).resolve().parent / "point_cloud_feature"


def cal_SHOT(A_points, A_normals, temp_dir, A_key_points, save_dir=None, radius=25.0):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    points_dir = temp_dir / "points.pcd"
    normal_dir = temp_dir / "normals.txt"
    key_points_dir = temp_dir / "key_points.pcd"
    feature_dir = Path(save_dir) if save_dir is not None else temp_dir / "SHOT_features.txt"

    txt2pcd(A_points, str(points_dir))
    np.savetxt(normal_dir, A_normals, fmt="%.5f")
    txt2pcd(A_key_points, str(key_points_dir))

    binary = _feature_binary_path()
    if not binary.exists():
        raise FileNotFoundError(
            f"未找到 SHOT 特征二进制: {binary}. "
            "请先运行 scripts/build_point_cloud_feature.sh。"
        )

    cmd = [
        str(binary),
        str(points_dir),
        str(normal_dir),
        str(key_points_dir),
        f"{radius:.2f}",
    ]
    with open(feature_dir, "w") as stdout_handle:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
        )

    features = read_features(str(feature_dir))
    valid_mask = np.ones(len(A_key_points), dtype=bool)
    stderr_text = result.stderr or ""
    invalid_indices = []
    for match in re.finditer(r"index\s+(\d+)", stderr_text):
        invalid_indices.append(int(match.group(1)))
    if invalid_indices:
        invalid_indices = sorted(set(i for i in invalid_indices if 0 <= i < len(valid_mask)))
        valid_mask[invalid_indices] = False

    if features.shape[0] != int(valid_mask.sum()):
        # Fallback: keep ordering stable even if the binary omits points without
        # reporting every invalid index on stderr.
        valid_mask[:] = False
        valid_mask[: features.shape[0]] = True

    if stderr_text.strip():
        print(stderr_text.strip())

    return features, valid_mask
