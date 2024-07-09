import torch
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix


def mse_to_psnr(mse):
    psnr = - 10 * torch.log(mse) / torch.log(torch.Tensor([10.]))
    return psnr


def pixel_acc(segm_pred, segm_gt):
    """
    :param segm_pred: (B, N, K) torch.Tensor.
    :param segm_gt: (B, N) torch.Tensor.
    :return:
    """
    segm_pred = torch.argmax(segm_pred, dim=-1)
    acc = (segm_pred == segm_gt).float().mean(1)
    return acc


def align_transl(pc1, pc2):
    """
    Solve a scaled rigid transformation from pc1 to pc2.
    :param pc1: (N, 3) torch.Tensor.
    :param pc2: (N, 3) torch.Tensor.
    :return:
        s: scalar () torch.Tensor.
        R: (3, 3) torch.Tensor.
        t: (3,) torch.Tensor.
    """
    pc1_mean = torch.mean(pc1, dim=0)
    pc1_centered = pc1 - pc1_mean
    pc2_mean = torch.mean(pc2, dim=0)
    pc2_centered = pc2 - pc2_mean

    S = torch.mm(pc1_centered.transpose(0, 1), pc2_centered)
    u, s, v = torch.svd(S, some=False, compute_uv=True)
    R = torch.mm(v, u.transpose(0, 1))
    det = torch.det(R)

    # Correct reflection matrix to rotation matrix
    diag = torch.ones(3)
    diag[2] = det
    R = v.mm(torch.diag_embed(diag).mm(u.transpose(0, 1)))

    # Solve scale factor
    rot_pc1_centered = torch.einsum('ij,nj->ni', R, pc1_centered)
    s = (pc2_centered * rot_pc1_centered).sum() / (rot_pc1_centered * rot_pc1_centered).sum()

    t = pc2_mean - s * torch.einsum('ij,j->i', R, pc1_mean)
    return s, R, t


def absolute_traj_error(pose1, pose2):
    """
    First align traj1 to traj2, then compute the error.
    :param pose1: (N, 4, 4) torch.Tensor.
    :param pose2: (N, 4, 4) torch.Tensor.
    """
    # Align by translation to find scaling factor
    rot1, transl1 = pose1[:, :3, :3], pose1[:, :3, 3]
    rot2, transl2 = pose2[:, :3, :3], pose2[:, :3, 3]
    s, R, t = align_transl(transl1, transl2)
    if torch.isnan(s).any() or torch.isnan(R).any() or torch.isnan(t).any():
        s, R, t = 1., torch.eye(3), torch.zeros(3)

    # Align by rotation and translation
    y, z = torch.Tensor([0., -1., 0.]), torch.Tensor([0., 0., -1.])
    y1, z1 = torch.einsum('nij,j->ni', rot1, y) + transl1, torch.einsum('nij,j->ni', rot1, z) + transl1
    pc1 = torch.cat([transl1, y1, z1], 0)
    y2, z2 = s * torch.einsum('nij,j->ni', rot2, y) + transl2, s * torch.einsum('nij,j->ni', rot2, z) + transl2
    pc2 = torch.cat([transl2, y2, z2], 0)
    s, R, t = align_transl(pc1, pc2)
    if torch.isnan(s).any() or torch.isnan(R).any() or torch.isnan(t).any():
        s, R, t = 1., torch.eye(3), torch.zeros(3)

    rot1_aligned = torch.einsum('ij,njk->nik', R, rot1)     # (N, 3, 3)
    transl1_aligned = s * torch.einsum('ij,nj->ni', R, transl1) + t     # (N, 3)

    # Rotation error
    rot1_aligned = matrix_to_quaternion(rot1_aligned)
    rot2 = matrix_to_quaternion(rot2)
    dist1 = (rot1_aligned - rot2).norm(dim=1)
    dist2 = (rot1_aligned + rot2).norm(dim=1)
    R_error = torch.minimum(dist1, dist2).mean()

    # Translation error
    t_error = (transl1_aligned - transl2).norm(dim=1).mean()
    return R_error, t_error, s, R, t


if __name__ == '__main__':
    from pytorch3d.transforms import random_rotation, random_rotations

    torch.manual_seed(0)

    """
    Check align function
    """
    gt_s = 0.2
    gt_R = random_rotation()
    gt_t = torch.randn(3)

    pc1 = torch.randn(10, 3)
    pc2 = gt_s * torch.einsum('ij,nj->ni', gt_R, pc1) + gt_t

    delta = 0.2 * torch.randn(pc2.shape)
    pc2 = pc2 + delta

    s, R, t = align_transl(pc1, pc2)


    """
    Check trajectory error function
    """
    gt_s = 0.2
    gt_R = random_rotation()
    gt_t = torch.randn(3)

    rot1 = random_rotations(10)
    transl1 = torch.randn(10, 3)
    rot2 = torch.einsum('ij,njk->nik', gt_R, rot1)
    transl2 = gt_s * torch.einsum('ij,nj->ni', gt_R, transl1) + gt_t

    R_error, t_error, s, R, t = absolute_traj_error(rot1, transl1, rot2, transl2)


print('Good')