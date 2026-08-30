from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from Sample import Sample
from extract_points.VoxEM import VoxEM
from extract_points.Supporting import load_sample_points
from alignment.Utils import load_xyz, load_sample, cal_SHOT, find_correspondences


def map_correspondence_coords_to_full_points(
    source_points,
    target_points,
    source_corr,
    target_corr,
):
    arrays = {
        "source_points": np.asarray(source_points),
        "target_points": np.asarray(target_points),
        "source_corr": np.asarray(source_corr),
        "target_corr": np.asarray(target_corr),
    }
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{name} must have shape (N,3), got {array.shape}")
    if arrays["source_corr"].shape[0] != arrays["target_corr"].shape[0]:
        raise ValueError("source_corr and target_corr must have the same length")

    source_distances, source_indices = cKDTree(arrays["source_points"]).query(
        arrays["source_corr"],
        k=1,
    )
    target_distances, target_indices = cKDTree(arrays["target_points"]).query(
        arrays["target_corr"],
        k=1,
    )
    indices = np.stack([source_indices, target_indices], axis=1).astype(np.int64)
    distances = np.stack([source_distances, target_distances], axis=1).astype(np.float64)
    return indices, distances


def sample_single_map(data_dir, src_name, src_contour, voxel_size, sample_only=False):
    mrc_file = f"{data_dir}/{src_name}"
    sample_file = f"{data_dir}/{src_name[:-4]}_{voxel_size:.2f}.txt"

    Sample(mrc_file, src_contour, voxel_size, 16.00, sample_file, 0)

    mrcobject = VoxEM()
    mrcobject.IO_ReadMrc(
        mrc_inputname=mrc_file,
        voxel_outputname="Default",
        description_outputname="Default",
        statistics_outputname="Default",
    )
    mrcobject.Voxel_Prune_RangeZero(
        lowerbound=src_contour,
        upperbound=None,
        inputname="Default",
        outputname="Voxel",
    )
    vox = mrcobject.voxel_workspace["Voxel"]
    start = mrcobject.description_workspace["Default"]["Start"]
    vox_length = mrcobject.description_workspace["Default"]["Angstrom"]
    vox_ang = np.array(vox_length).astype(np.float32) / np.array(vox.shape).astype(np.float32)

    sample_points, _ = load_sample_points(sample_file)
    shifting = start * vox_ang
    sample_points = sample_points - shifting

    mrcobject.point_workspace["sample"] = sample_points.T
    print("sample points:", sample_points.shape)

    if sample_only:
        print("仅均匀采样：跳过 Meanshift/DBSCAN 与关键点导出")
        return

    mrcobject.Point_Create_Meanshift_sample(
        lower_bound=src_contour,
        window=17,
        voxel_inputname="Voxel",
        bandwidth=3.0,
        point_outputname="Meanshift",
        point_inputname="sample",
        iteration=2000,
        convergence=0.000187,
        step_size=0.05,
    )
    mrcobject.Point_Transform_DBSCAN(
        distance_tolerance=voxel_size * 3,
        clustersize_tolerance=3,
        inputname="Meanshift",
        outputname="Meanshift",
    )
    print("meanshift points:", mrcobject.point_workspace["Meanshift"].shape)

    mrcobject.Point_Transform_DBSCAN(
        distance_tolerance=voxel_size,
        clustersize_tolerance=1,
        inputname="Meanshift",
        outputname="DBSCAN",
        centroid_only=True,
    )
    print("DBSCAN points:", mrcobject.point_workspace["DBSCAN"].shape)

    src_stem = Path(src_name).stem
    keypoint_xyz_file = f"{data_dir}/Points_{src_stem}_Key.xyz"
    mrcobject.IO_WriteXYZ(
        point_inputname="DBSCAN",
        file_outputname=keypoint_xyz_file,
        atom_name="H",
        shifting=shifting,
    )

    keypoint_coords = mrcobject.point_workspace["DBSCAN"].T
    np.save(f"{data_dir}/{src_name[:-4]}_keypoints.npy", keypoint_coords)
    print(f"Saved keypoint coordinates to {data_dir}/{src_name[:-4]}_keypoints.npy")


def main():
    parser = ArgumentParser(description="Sample two density maps and build keypoint correspondences.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing source/target MRC files")
    parser.add_argument("--source", type=str, required=True, help="Source MRC filename")
    parser.add_argument("--target", type=str, required=True, help="Target MRC filename")
    parser.add_argument("--source_contour", type=float, default=0.267, help="Recommended source contour level")
    parser.add_argument("--target_contour", type=float, default=0.267, help="Recommended target contour level")
    parser.add_argument("--voxel", type=float, default=3.0, help="Sampling voxel size")
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Only perform uniform sampling and save points/normals; skip clustering and matching",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    source_name = args.source
    target_name = args.target

    sample_single_map(data_dir, source_name, args.source_contour, args.voxel, args.sample_only)
    sample_single_map(data_dir, target_name, args.target_contour, args.voxel, args.sample_only)

    source_sample_dir = f"{data_dir}/{source_name[:-4]}_{args.voxel:.2f}.txt"
    target_sample_dir = f"{data_dir}/{target_name[:-4]}_{args.voxel:.2f}.txt"

    sample_A_points, sample_A_normals = load_sample(source_sample_dir)
    sample_B_points, sample_B_normals = load_sample(target_sample_dir)

    src_base = source_name[:-4]
    tgt_base = target_name[:-4]
    np.save(f"{data_dir}/{src_base}_{args.voxel:.2f}_points.npy", sample_A_points)
    np.save(f"{data_dir}/{src_base}_{args.voxel:.2f}_normals.npy", sample_A_normals)
    np.save(f"{data_dir}/{tgt_base}_{args.voxel:.2f}_points.npy", sample_B_points)
    np.save(f"{data_dir}/{tgt_base}_{args.voxel:.2f}_normals.npy", sample_B_normals)

    if args.sample_only:
        print(f"已保存均匀采样点云到 {data_dir}，跳过聚类与配准。")
        return

    source_key_dir = f"{data_dir}/Points_{Path(source_name).stem}_Key.xyz"
    target_key_dir = f"{data_dir}/Points_{Path(target_name).stem}_Key.xyz"

    A_keypoint = np.asarray(load_xyz(source_key_dir).points)
    B_keypoint = np.asarray(load_xyz(target_key_dir).points)
    np.save(f"{data_dir}/{src_base}_keypoints.npy", A_keypoint)
    np.save(f"{data_dir}/{tgt_base}_keypoints.npy", B_keypoint)

    temp_dir = Path(data_dir) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    A_key_feats, A_valid_mask = cal_SHOT(sample_A_points, sample_A_normals, temp_dir, A_keypoint, radius=args.voxel * 7)
    B_key_feats, B_valid_mask = cal_SHOT(sample_B_points, sample_B_normals, temp_dir, B_keypoint, radius=args.voxel * 7)

    A_keypoint = A_keypoint[A_valid_mask]
    B_keypoint = B_keypoint[B_valid_mask]
    print(f"Valid SHOT keypoints: source={len(A_keypoint)} target={len(B_keypoint)}")

    corrs_A, corrs_B = find_correspondences(A_key_feats, B_key_feats, mutual_filter=True)
    A_corr = A_keypoint[corrs_A, :].T
    B_corr = B_keypoint[corrs_B, :].T
    print(f"generates {A_corr.shape[1]} putative correspondences.")

    corr_indices = np.stack([corrs_A, corrs_B], axis=1)
    np.save(f"{data_dir}/corr_indices_{src_base}_{tgt_base}.npy", corr_indices)
    np.save(f"{data_dir}/corr_A_coords_{src_base}_{tgt_base}.npy", A_corr.T)
    np.save(f"{data_dir}/corr_B_coords_{src_base}_{tgt_base}.npy", B_corr.T)
    full_corr_indices, full_corr_distances = map_correspondence_coords_to_full_points(
        sample_A_points,
        sample_B_points,
        A_corr.T,
        B_corr.T,
    )
    np.save(
        f"{data_dir}/corr_full_indices_{src_base}_{tgt_base}.npy",
        full_corr_indices,
    )
    np.save(
        f"{data_dir}/corr_full_mapping_distances_{src_base}_{tgt_base}.npy",
        full_corr_distances,
    )
    print(
        "Mapped keypoint correspondences to full sampled clouds: "
        f"source max={full_corr_distances[:, 0].max():.6f} Å, "
        f"target max={full_corr_distances[:, 1].max():.6f} Å"
    )
    print(f"Saved npy files to {data_dir}")


if __name__ == "__main__":
    main()
