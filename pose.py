import torch
import torch.nn as nn
from pytorch3d.transforms import se3_exp_map, se3_log_map


def pose_inv(pose):
    """
    :param pose: (Nv, 4, 4) torch.Tensor.
    :return:
        pose_inv: (Nv, 4, 4) torch.Tensor.
    """
    pose_shape = pose.shape
    pose = pose.reshape(-1, 4, 4)
    pose_inv = pose.clone()
    pose_inv[:, :3, :3] = pose[:, :3, :3].transpose(1, 2)
    pose_inv[:, :3, 3] = -torch.einsum('vij,vj->vi', pose_inv[:, :3, :3], pose[:, :3, 3])
    pose_inv = pose_inv.reshape(pose_shape)
    return pose_inv


def compose_cam_pose(cam_pose, obj_pose):
    """
    :param cam_pose: (Nv, 4, 4) torch.Tensor.
    :param obj_pose: (Nv, K, 4, 4) torch.Tensor.
    :return:
        cam2obj_pose: (Nv, K, 4, 4) torch.Tensor.
    """
    # obj_pose_inv = torch.linalg.inv(obj_pose)
    obj_pose_inv = pose_inv(obj_pose)
    cam2obj_pose = torch.einsum('nkij,njm->nkim', obj_pose_inv, cam_pose)
    return cam2obj_pose


def decompose_cam_pose(cam_pose, cam2obj_pose):
    """
    :param cam_pose: (Nv, 4, 4) torch.Tensor.
    :param cam2obj_pose: (Nv, K, 4, 4) torch.Tensor.
    :return:
        obj_pose: (Nv, K, 4, 4) torch.Tensor.
    """
    # cam2obj_pose_inv = torch.linalg.inv(cam2obj_pose)
    cam2obj_pose_inv = pose_inv(cam2obj_pose)
    obj_pose = torch.einsum('nij,nkjm->nkim', cam_pose, cam2obj_pose_inv)
    return obj_pose


class LiePose(nn.Module):
    def __init__(self, n_view):
        super().__init__()
        self.se3 = torch.nn.Embedding(n_view, 6)    # refinement
        nn.init.zeros_(self.se3.weight)

        SE3 = torch.eye(4).reshape(1, 4, 4).repeat(n_view, 1, 1)    # base
        self.SE3 = nn.Parameter(SE3, requires_grad=False)

    def get_pose(self, vid):
        """
        :param vid: int.
        :return:
            pose: (4, 4) torch.Tensor.
        """
        se3 = self.se3.weight[vid:(vid+1)]
        pose = se3_exp_map(se3)[0]
        pose = pose.transpose(-1, -2)

        SE3 = self.SE3[vid]
        pose = torch.matmul(pose, SE3)
        return pose

    def get_poses(self, vid_list):
        """
        :param vid_list: list of [int, ...].
        :return:
            poses: (N, 4, 4) torch.Tensor.
        """
        poses = [self.get_pose(vid) for vid in vid_list]
        poses = torch.stack(poses, 0)
        return poses

    def get_all_poses(self):
        """
        :return:
            poses: (N, 4, 4) torch.Tensor.
        """
        se3 = self.se3.weight
        poses = se3_exp_map(se3)
        poses = poses.transpose(-1, -2)

        poses = torch.bmm(poses, self.SE3)
        return poses

    def load_base_poses(self, poses):
        """
        :param poses: (N, 4, 4) torch.Tensor.
        """
        self.SE3.data = poses


def project_points(points, img_h, img_w, focal, poses):
    """
    :param points: (K, Nr, 3) torch.Tensor.
    :param poses: (K, 4, 4) torch.Tensor.
    :return:
        uv: (K, Nr, 2) torch.Tensor.
    """
    R = poses[:, :3, :3].transpose(1, 2)
    t = - torch.einsum('kij,kj->ki', R, poses[:, :3, 3])
    uvd = torch.einsum('kij,knj->kni', R, points) + t.unsqueeze(1)      # (K, Nr, 3)

    u, v = uvd[..., 0] / (-uvd[..., 2]), uvd[..., 1] / (-uvd[..., 2])
    u = u * focal + 0.5 * img_w
    v = - v * focal + 0.5 * img_h
    uv = torch.stack([u, v], -1)
    return uv


def project_flow(depth_map, rays_o, rays_d, img_h, img_w, focal, neighbor_poses, uv):
    """
    :param depth_map: (K, Nr) torch.Tensor.
    :param rays_o: (K, Nr, 3) torch.Tensor.
    :param rays_d: (K, Nr, 3) torch.Tensor.
    :param neighbor_poses: (K, 4, 4) torch.Tensor.
    :param uv: (K, Nr, 2) torch.Tensor.
    :return:
        flow: (K, Nr, 2) torch.Tensor.
    """
    points = rays_o + depth_map.unsqueeze(2) * rays_d
    uv_warp = project_points(points, img_h, img_w, focal, neighbor_poses)
    flows = uv_warp - uv
    return flows


print('Good')