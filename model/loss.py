from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def unify_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()
    else:
        X = points
        if lengths is None:
            lengths = torch.full((X.shape[0],), X.shape[1], dtype=torch.int64, device=X.device)
    return X, lengths, normals


def compute_truncated_chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    trunc=0.2,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    x, x_lengths, x_normals = unify_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = unify_pointcloud_input(y, y_lengths, y_normals)
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]
    cham_y = y_nn.dists[..., 0]


    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    cham_x = torch.sqrt(cham_x).sum(1)
    cham_y = torch.sqrt(cham_y).sum(1)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1) 
        cham_norm_y = cham_norm_y.sum(1)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y

    return cham_dist


def compute_nn_chamfer_l2(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    return_per_direction: bool = False,
):
    """
    计算最近邻 Chamfer L2 距离（无截断版本）。

    Args:
        x (Tensor | Pointclouds): 源点云或 Pytorch3D Pointclouds
        y (Tensor | Pointclouds): 目标点云或 Pytorch3D Pointclouds
        x_lengths (Tensor | None): 源点云每个 batch 的有效点数
        y_lengths (Tensor | None): 目标点云每个 batch 的有效点数
        weights (Tensor | None): batch 级别权重
        batch_reduction (str | None): 对 batch 汇聚的方式，支持 "mean"/"sum"/None
        point_reduction (str): 对点汇聚的方式，当前支持 "mean"（默认）或 "sum"
        return_per_direction (bool): 是否返回双向距离
    """
    x, x_lengths, _ = unify_pointcloud_input(x, x_lengths, None)
    y, y_lengths, _ = unify_pointcloud_input(y, y_lengths, None)

    N = x.shape[0]

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1).dists[..., 0]
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1).dists[..., 0]

    cham_x = torch.sqrt(x_nn)
    cham_y = torch.sqrt(y_nn)

    if point_reduction == "mean":
        cham_x = cham_x.sum(1) / x_lengths
        cham_y = cham_y.sum(1) / y_lengths
    elif point_reduction == "sum":
        cham_x = cham_x.sum(1)
        cham_y = cham_y.sum(1)
    else:
        raise ValueError(f"Unsupported point_reduction '{point_reduction}', expect 'mean' or 'sum'.")

    if weights is not None:
        weights = weights.view(N)
        cham_x = cham_x * weights
        cham_y = cham_y * weights

    if batch_reduction is not None:
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x = cham_x / div
            cham_y = cham_y / div
        elif batch_reduction != "sum":
            raise ValueError(f"Unsupported batch_reduction '{batch_reduction}', expect 'mean', 'sum' or None.")

    if return_per_direction:
        return cham_x, cham_y
    return cham_x + cham_y


def _ensure_batched_points(points: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if points.dim() == 2:
        return points.unsqueeze(0), True
    if points.dim() == 3:
        return points, False
    raise ValueError(f"Expected point cloud shape (N, D) or (B, N, D), got {tuple(points.shape)}")


def _uniform_log_weights(points: torch.Tensor) -> torch.Tensor:
    batch_size, n_points, _ = points.shape
    log_weight = -torch.log(points.new_tensor(float(n_points)))
    return log_weight.expand(batch_size, n_points).clone()


def _sinkhorn_cost_matrix(x: torch.Tensor, y: torch.Tensor, p: int) -> torch.Tensor:
    dist = torch.cdist(x, y, p=2).clamp_min(0.0)
    if p == 1:
        return dist
    if p == 2:
        return 0.5 * dist.pow(2)
    return dist.clamp_min(1e-12).pow(p) / p


def _sinkhorn_softmin(cost: torch.Tensor, log_w: torch.Tensor, eps: float, dim: int) -> torch.Tensor:
    return -eps * torch.logsumexp(log_w - cost / eps, dim=dim)


def _sinkhorn_potentials(
    x: torch.Tensor,
    y: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    p: int,
    n_iters: int,
    tol: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cost = _sinkhorn_cost_matrix(x, y, p)
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    with torch.no_grad():
        for _ in range(n_iters):
            f_prev = f
            f = _sinkhorn_softmin(cost - g.unsqueeze(1), log_b.unsqueeze(1), eps, dim=2)
            g = _sinkhorn_softmin(cost.transpose(1, 2) - f.unsqueeze(1), log_a.unsqueeze(1), eps, dim=2)
            if tol > 0.0 and (f - f_prev).abs().max().item() < tol:
                break

    f = _sinkhorn_softmin(cost - g.unsqueeze(1), log_b.unsqueeze(1), eps, dim=2)
    g = _sinkhorn_softmin(cost.transpose(1, 2) - f.unsqueeze(1), log_a.unsqueeze(1), eps, dim=2)
    return f, g


def _regularized_ot_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    p: int,
    n_iters: int,
    tol: float,
) -> torch.Tensor:
    f, g = _sinkhorn_potentials(x, y, log_a, log_b, eps, p, n_iters, tol)
    return (log_a.exp() * f).sum(dim=1) + (log_b.exp() * g).sum(dim=1)


def sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    blur: float = 0.1,
    p: int = 2,
    n_iters: int = 100,
    tol: float = 1e-3,
    debias: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    x, was_2d = _ensure_batched_points(x)
    y, _ = _ensure_batched_points(y)

    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"x and y must share batch size and dimension, got {tuple(x.shape)} and {tuple(y.shape)}")
    if blur <= 0:
        raise ValueError("blur must be positive.")
    if p < 1:
        raise ValueError("p must be >= 1.")

    log_a = torch.log(a.clamp_min(1e-12)) if a is not None else _uniform_log_weights(x)
    log_b = torch.log(b.clamp_min(1e-12)) if b is not None else _uniform_log_weights(y)
    if log_a.dim() == 1:
        log_a = log_a.unsqueeze(0)
    if log_b.dim() == 1:
        log_b = log_b.unsqueeze(0)
    if log_a.shape != x.shape[:2] or log_b.shape != y.shape[:2]:
        raise ValueError("Weights must have shape (N,) or (B, N) matching their point clouds.")

    log_a = log_a - torch.logsumexp(log_a, dim=1, keepdim=True)
    log_b = log_b - torch.logsumexp(log_b, dim=1, keepdim=True)
    eps = float(blur) ** p

    cost = _regularized_ot_cost(x, y, log_a, log_b, eps, p, n_iters, tol)
    if debias:
        cost_xx = _regularized_ot_cost(x, x, log_a, log_a, eps, p, n_iters, tol)
        cost_yy = _regularized_ot_cost(y, y, log_b, log_b, eps, p, n_iters, tol)
        cost = cost - 0.5 * cost_xx - 0.5 * cost_yy

    if reduction == "none":
        return cost[0] if was_2d else cost
    if reduction == "sum":
        return cost.sum()
    if reduction == "mean":
        return cost.mean()
    raise ValueError(f"Unsupported reduction '{reduction}', expect 'mean', 'sum' or 'none'.")


def landmark_cost(x, y):
    return torch.mean(torch.sum((x - y) ** 2, dim=-1))


# ===== KNN 工具与损失 =====
def precompute_knn_indices(points_t: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(points_t, points_t, p=2)
    _, knn_idx = torch.topk(dists, k=k+1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]
    return knn_idx


def knn_motion_consistency_loss(
    warped: torch.Tensor,
    original: torch.Tensor,
    knn_idx: torch.Tensor,
    w_knn_motion: float = 1.0,
) -> torch.Tensor:
    if w_knn_motion <= 0:
        return warped.new_tensor(0.0)
    disp = warped - original
    disp = disp 
    neigh_disp = disp[knn_idx] 
    mean_neigh = neigh_disp.mean(dim=1)
    lap = disp - mean_neigh
    loss = (lap * lap).sum(dim=1).mean() * w_knn_motion * 100.0 #消除网络中scale = 0.01对运动的影响
    return loss
