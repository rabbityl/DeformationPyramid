from typing import Union
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import numpy as np

def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def point_2_plane_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None

):
    '''
    '''

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    N, P1, D = x.shape
    assert N==1

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    x_ref_normal = y_normals[0][x_nn.idx[0,:, 0]]
    x_ref_point = y[0][x_nn.idx[0,:, 0] ]

    y_ref_normal = x_normals[0][y_nn.idx[0,:, 0]]
    y_ref_point = x[0][y_nn.idx[0,:, 0] ]

    x_2_plane = ( (x[0] - x_ref_point ) * x_ref_normal ).pow(2).sum(1).sqrt().mean()
    y_2_plane = ( (y[0] - y_ref_point ) * y_ref_normal ).pow(2).sum(1).sqrt().mean()

    p2plane = x_2_plane + y_2_plane

    return p2plane, x_2_plane, y_2_plane, x_ref_normal

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
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
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

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # truncation
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
        # Gather the normals using the indices and keep only value for k=0
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

    # Apply point reduction
      
    #cham_x = cham_x.sum(1)  # (N,)
    #cham_y = cham_y.sum(1)  # (N,)

    # use l1 norm, more robust to partial case
    cham_x = torch.sqrt(cham_x).sum(1)  # (N,)
    cham_y = torch.sqrt(cham_y).sum(1)  # (N,)

    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
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
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist


def arap_cost (R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R [:, None]
    g_i = g [:, None]
    t_i = t [:, None]

    g_j = g [e]
    t_j = t [e]

    # if lietorch :
    #     e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)

    e_ij = (((R_i @ (g_j - g_i)[...,None]).squeeze() + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)

    o = (w * e_ij ).mean()

    return o


def silhouette_cost(x, y, renderer):

    INF = 1e+6

    px, dx = renderer(x)
    py, dy = renderer(y)

    px, dx, py, dy = map ( lambda feat: feat.squeeze(), [px, dx, py, dy])


    dx[dx<0] = INF
    dy[dy<0] = INF

    dx,_ = torch.min(dx, dim=-1,)
    dy,_ = torch.min(dy, dim=-1)

    dx[dx==INF] = 0
    dy[dy==INF] = 0

    x_mask = px[...,0] > 0
    y_mask = py[...,0] > 0


    # plt.imshow(py.detach().cpu().numpy())
    # plt.show()
    #
    # # plt.figure(figsize=(10, 10))
    # plt.imshow(dx.detach().cpu().numpy())
    # plt.show()
    # # #
    # plt.imshow(dy.detach().cpu().numpy())
    # plt.show()


    depth_error = (dx - dy)**2
    #
    # plt.imshow(depth_error.detach().cpu().numpy())
    # plt.show()


# depth_error[depth_error>0.01] = 0

    silh_error = (px - py)**2


    silh_error = silh_error[~y_mask]

    depth_error = depth_error [ y_mask * x_mask]

    depth_error[ depth_error > 0.06**2] = 0
    # plt.imshow(depth_error.detach().cpu().numpy())
    # plt.show()

    silh_loss = torch.mean( silh_error)
    depth_loss = torch.mean( depth_error )


    return silh_loss + depth_loss


def landmark_cost(x, y):
    loss = torch.mean(
        torch.sum( (x-y)**2, dim=-1 ))
    return loss


def chamfer_dist(src_pcd,   tgt_pcd, samples=1000):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src[:samples]]
    t_sample = tgt_pcd[ tgt[:samples]]
    cham_dist = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=1e+10)

    return cham_dist


def nerfies_regularization( jacobian, eps=1e-6):
    jacobian=jacobian.cpu().double()
    svals = jacobian.svd(compute_uv=False).S # small SVD runs faster on cpu
    svals[svals<eps] = eps
    log_svals = torch.log( svals.max(dim=1)[0] )
    loss = torch.mean( log_svals**2 )
    return loss


def scene_flow_metrics(pred, labels, strict=0.025, relax = 0.05):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 1)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: AccS
    error_lt_5 = torch.BoolTensor((l2_norm < strict))
    relative_err_lt_5 = torch.BoolTensor((relative_err < strict))
    AccS = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: AccR
    error_lt_10 = torch.BoolTensor((l2_norm < relax))
    relative_err_lt_10 = torch.BoolTensor((relative_err < relax))
    AccR = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    relative_err_lt_30 = torch.BoolTensor(relative_err > 0.3)
    outlier = torch.mean(relative_err_lt_30.float()).item()

    return EPE3D*100, AccS*100, AccR*100, outlier*100



def scene_flow_EPE_np(pred, labels, mask):
    '''
    :param pred: [B, N, 3]
    :param labels:
    :param mask: [B, N]
    :return:
    '''
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)  , (error/gtflow_len <= 0.05)  ), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)  , (error/gtflow_len <= 0.1)  ), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.sum(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.sum(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.sum(EPE)
    return EPE, acc1, acc2


def compute_flow_metrics( flow, flow_gt, overlap=None):

    metric_info = {}

    # full point cloud
    epe, AccS, AccR, outlier = scene_flow_metrics(flow, flow_gt)
    metric_info.update(
        {
            "full-epe": epe,
            "full-AccS": AccS,
            "full-AccR": AccR,
            "full-outlier": outlier
        }
    )

    if overlap is not None:

        # visible
        epe, AccS, AccR, outlier = scene_flow_metrics(flow[overlap], flow_gt[overlap])
        metric_info.update(
            {
                "vis-epe": epe,
                "vis-AccS": AccS,
                "vis-AccR": AccR,
                "vis-outlier": outlier

            }
        )

        # occluded
        epe, AccS, AccR, outlier = scene_flow_metrics(flow[~overlap], flow_gt[~overlap])
        metric_info.update(
            {
                "occ-epe": epe,
                "occ-AccS": AccS,
                "occ-AccR": AccR,
                "occ-outlier": outlier
            }
        )

    return metric_info
