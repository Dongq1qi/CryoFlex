#使用KNN约束相邻点有相同的运动趋势，以及前几层训练使用corr_loss

import numpy as np
import torch
import argparse
import os
import random
from model.loss import (
    compute_truncated_chamfer_distance,
    landmark_cost,
    precompute_knn_indices,
    knn_motion_consistency_loss,
    sinkhorn_loss,
)
from model.net import Deformation_Pyramid
from easydict import EasyDict as edict
import torch.nn as nn
BCE = nn.BCELoss()

## KNN 工具与损失已移动到 model.loss


# ---------------------------

def normalize_points(points):
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    return points, center, scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="源点云 .npy 文件路径")
    parser.add_argument("--target", type=str, required=True, help="目标点云 .npy 文件路径")
    parser.add_argument("--corr", type=str, required=False, default=None, help="对应关系文件路径（支持 .txt 或 .npy，形状[N,2]），可选")
    parser.add_argument("--output", type=str, required=True, help="最终保存结果 .npy 路径")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--iters", type=int, default=None, help="每层优化迭代数；默认使用脚本内配置")
    parser.add_argument("--w_sinkhorn", type=float, default=None, help="Sinkhorn/transport loss 权重；默认使用脚本内配置")
    parser.add_argument("--w_knn_motion", type=float, default=None, help="KNN local motion coherence 权重；默认使用脚本内配置")
    parser.add_argument("--w_ldmk", type=float, default=None, help="correspondence anchor loss 权重；默认使用脚本内配置")
    parser.add_argument("--corr_loss_levels", type=int, default=None, help="前多少层使用 corr loss；默认使用脚本内配置")
    parser.add_argument("--gpu_mode", type=int, choices=[0, 1], default=None, help="1 使用 GPU，0 使用 CPU；默认使用脚本内配置")
    args = parser.parse_args()
    config = {
        "gpu_mode": True,
        "iters": 500,
        "lr": 0.01,
        "max_break_count": 30,
        "break_threshold_ratio": 0.0001,
        "motion_type": "SE3",       #[ "Sim3", "SE3", "sflow"]
        "rotation_format": "euler",    #[ "axis_angle", "euler", "quaternion", "6D"]
         "m": 9,
        "k0": -5,
        "depth": 3,
        "width": 128,
        "act_fn": "relu",
        "w_reg": 0,
        "w_ldmk": 1,
        "w_cd": 1,
        "w_sinkhorn": 1000,
        "sinkhorn_blur": 0.1,
        "sinkhorn_p": 2,
        "sinkhorn_iters": 100,
        "sinkhorn_tol": 1e-3,
        "sinkhorn_debias": True,
        "corr_loss_levels": 2,        # 仅在前几层使用 corr_loss

        # ===== 新增：KNN 约束配置 =====
        "knn_k": 32,                  # KNN 邻居数
        "w_knn_motion": 1,         # 邻域"运动一致性"权重（位移平滑/一致）
    }

    config = edict(config)
    if args.iters is not None:
        config.iters = args.iters
    if args.w_sinkhorn is not None:
        config.w_sinkhorn = args.w_sinkhorn
    if args.w_knn_motion is not None:
        config.w_knn_motion = args.w_knn_motion
    if args.w_ldmk is not None:
        config.w_ldmk = args.w_ldmk
    if args.corr_loss_levels is not None:
        config.corr_loss_levels = args.corr_loss_levels
    if args.gpu_mode is not None:
        config.gpu_mode = bool(args.gpu_mode)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(
        "配置: "
        f"seed={args.seed}, iters={config.iters}, w_cd={config.w_cd}, "
        f"w_sinkhorn={config.w_sinkhorn}, w_ldmk={config.w_ldmk}, "
        f"corr_loss_levels={config.corr_loss_levels}, "
        f"w_knn_motion={config.w_knn_motion}, gpu_mode={config.gpu_mode}"
    )
    source = args.source
    target = args.target

    # 仅在提供 corr 文件时加载对应关系
    A_indices = None
    B_indices = None
    if args.corr is not None:
        corr_path = args.corr
        ext = os.path.splitext(corr_path)[1].lower()
        if ext == ".npy":
            corr_matrix = np.load(corr_path)
        else:
            corr_matrix = np.loadtxt(corr_path, dtype=int)
        corr_matrix = np.asarray(corr_matrix, dtype=int)
        if corr_matrix.ndim != 2 or corr_matrix.shape[1] != 2:
            raise ValueError(f"--corr 文件形状应为 [N,2]，实际为 {corr_matrix.shape}")
        A_indices = corr_matrix[:, 0]  # 第一列是源点云索引
        B_indices = corr_matrix[:, 1]  # 第二列是目标点云索引
        correspondences = np.stack([A_indices, B_indices], axis=1)
        print(f"加载对应关系：{len(A_indices)} 对")
    else:
        print("未提供对应关系文件，将不使用 corr_loss")


    source_points = np.load(source)
    target_points = np.load(target)

    N = source_points.shape[0]
    source_indices = np.arange(N)
    target_indices = np.arange(N)

    source_points, source_center, source_scale = normalize_points(source_points)
    #target_points = (target_points - source_center) / source_scale
    target_points, target_center, target_scale = normalize_points(target_points)

    device = torch.device("cuda" if (torch.cuda.is_available() and config.gpu_mode) else "cpu")
    source_points = torch.from_numpy(source_points).float().to(device)
    target_points = torch.from_numpy(target_points).float().to(device)
    source_indices = torch.from_numpy(source_indices).to(device)
    target_indices = torch.from_numpy(target_indices).to(device)
    
    # 将对应关系索引转换为 tensor 并保持为 long 类型（整数）
    if A_indices is not None and B_indices is not None:
        A_indices = torch.from_numpy(A_indices).long().to(device)
        B_indices = torch.from_numpy(B_indices).long().to(device)

    # 保留一份“原始(归一化后)源点云”用于位移计算（运动一致性基于原始位置）
    source_points_original = source_points.clone().detach()

    print(f"target: {target}")

    initial_cd_distance = compute_truncated_chamfer_distance(source_points[None], target_points[None], trunc=1e+9).item()
    print(f"初始CD距离: {initial_cd_distance:.6f}")
    if config.w_sinkhorn > 0:
        initial_sinkhorn_distance = sinkhorn_loss(
            source_points,
            target_points,
            blur=config.sinkhorn_blur,
            p=config.sinkhorn_p,
            n_iters=config.sinkhorn_iters,
            tol=config.sinkhorn_tol,
            debias=config.sinkhorn_debias,
        ).item()
        print(f"初始Sinkhorn距离: {initial_sinkhorn_distance:.6f}")

    # ===== 新增：预计算 KNN 索引（固定在原始源点云上） =====
    knn_idx = precompute_knn_indices(source_points_original, k=config.knn_k)  # [N, k]

    NDP = Deformation_Pyramid(
        depth=config.depth,
        width=config.width,
        device=device,
        k0=config.k0,
        m=config.m,
        nonrigidity_est=config.w_reg > 0,
        rotation_format=config.rotation_format,
        motion=config.motion_type,
    )

    for level in range(NDP.n_hierarchy):
        NDP.gradient_setup(optimized_level=level)
        optimizer = torch.optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)
        break_counter = 0
        loss_prev = 1e+6

        print(f"\n=== Starting Level {level} ===")

        for iter in range(config.iters):
            s_sample_warped, data = NDP.warp(source_points, max_level=level, min_level=level)
            cd_loss = compute_truncated_chamfer_distance(s_sample_warped[None], target_points[None], trunc=1e+9) 
            if config.w_sinkhorn > 0:
                sinkhorn_dist = sinkhorn_loss(
                    s_sample_warped,
                    target_points,
                    blur=config.sinkhorn_blur,
                    p=config.sinkhorn_p,
                    n_iters=config.sinkhorn_iters,
                    tol=config.sinkhorn_tol,
                    debias=config.sinkhorn_debias,
                )
            else:
                sinkhorn_dist = torch.tensor(0.0, device=device)
            corr_loss = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)
            
            # ===== KNN 运动一致性约束（总是计算） =====
            knn_motion_loss = knn_motion_consistency_loss(
                warped=s_sample_warped,
                original=source_points_original,
                knn_idx=knn_idx,
                w_knn_motion=config.w_knn_motion
            )
            
            #=== corr_loss ====
            # 仅在前几层使用 corr_loss（且提供了对应关系）
            if A_indices is not None and B_indices is not None and level < config.corr_loss_levels:
                corr_loss = landmark_cost(s_sample_warped[A_indices], target_points[B_indices])
            else:
                corr_loss = torch.tensor(0.0, device=device)

            # ===== reg_loss 计算（与test_data一致） =====
            if level > 0 and config.w_reg > 0:
                if (data is not None) and (level in data):
                    nonrigidity = data[level][1]
                    target_reg = torch.zeros_like(nonrigidity)
                    reg_loss = BCE(nonrigidity, target_reg)

            # ===== 基础损失计算（CD + Sinkhorn + KNN Motion参与反向传播） =====
            loss = (
                cd_loss * config.w_cd
                + sinkhorn_dist * config.w_sinkhorn
                + knn_motion_loss
                + corr_loss * config.w_ldmk
                + reg_loss * config.w_reg
            )

            if level > 0 and config.w_reg > 0:
                print(f"Level {level}, Iter {iter}: Total Loss = {loss.item():.6f}, CD Loss = {cd_loss.item():.6f}, Sinkhorn Loss = {sinkhorn_dist.item():.6f}, Reg Loss = {reg_loss.item():.6f}, KNN_motion = {knn_motion_loss.item():.6f}, Corr Loss = {corr_loss.item():.6f}")
            else:
                print(f"Level {level}, Iter {iter}: Loss = {loss.item():.6f}, CD Loss = {cd_loss.item():.6f}, Sinkhorn Loss = {sinkhorn_dist.item():.6f}, KNN_motion = {knn_motion_loss.item():.6f}, Corr Loss = {corr_loss.item():.6f}")
                pass

            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 层结束评估
        with torch.no_grad():
            level_cd_distance = compute_truncated_chamfer_distance(s_sample_warped[None], target_points[None], trunc=1e+9).item()
            print(f"Level {level} CD距离: {level_cd_distance:.6f}")
            if config.w_sinkhorn > 0:
                level_sinkhorn_distance = sinkhorn_loss(
                    s_sample_warped,
                    target_points,
                    blur=config.sinkhorn_blur,
                    p=config.sinkhorn_p,
                    n_iters=config.sinkhorn_iters,
                    tol=config.sinkhorn_tol,
                    debias=config.sinkhorn_debias,
                ).item()
                print(f"Level {level} Sinkhorn距离: {level_sinkhorn_distance:.6f}")


        # 下一层输入
        source_points = s_sample_warped.detach()

    # 反归一化：恢复到原始尺度
    final_points = source_points.detach().cpu().numpy()
    final_points = final_points * target_scale + target_center

    # 保存反归一化后的最终预测点云
    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, final_points)

if __name__ == "__main__":
    main()
