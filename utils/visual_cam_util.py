import numpy as np
import open3d as o3d


class Camera:
    def __init__(self, img_h, img_w, focal, pose):
        """
        :param pose: (4, 4) torch.Tensor.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.focal = focal
        self.pose = pose

    def get_uv(self):
        """
        :return:
            uv: (H*W, 2).
        """
        u, v = np.meshgrid(np.arange(self.img_w, dtype=np.float32), np.arange(self.img_h, dtype=np.float32),
                           indexing='xy')
        uv = np.stack((u, v), -1)
        uv = uv.reshape(-1, 2)
        return uv

    def back_project(self, depth, uv=None):
        """
        :param depth: (N,).
        :param uv: (N, 2).
        :return:
            pc_world: (N, 3).
        """
        if uv is None:
            uv = self.get_uv()

        u = (uv[:, 0] - 0.5 * self.img_w) / self.focal
        v = - (uv[:, 1] - 0.5 * self.img_h) / self.focal
        uv = np.stack((u, v), -1)
        uvd = np.concatenate((uv, -np.ones([depth.shape[0], 1])), -1)

        pc_cam = uvd * np.expand_dims(depth, 1)
        cam_rot, cam_transl = self.pose[:3, :3], self.pose[:3, 3]
        pc_world = np.einsum('ij,nj->ni', cam_rot, pc_cam) + cam_transl
        return pc_world


class BatchCameras:
    def __init__(self, img_h, img_w, focal, poses):
        """
        :param pose: (K, 4, 4) torch.Tensor.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.focal = focal
        self.poses = poses

    def get_uv(self):
        """
        :return:
            uv: (H*W, 2).
        """
        u, v = np.meshgrid(np.arange(self.img_w, dtype=np.float32), np.arange(self.img_h, dtype=np.float32),
                           indexing='xy')
        uv = np.stack((u, v), -1)
        uv = uv.reshape(-1, 2)
        return uv

    def back_project(self, depth, uv=None):
        """
        :param depth: (K, N).
        :param uv: (N, 2).
        :return:
            pc_world: (N, 3).
        """
        if uv is None:
            uv = self.get_uv()

        u = (uv[:, 0] - 0.5 * self.img_w) / self.focal
        v = - (uv[:, 1] - 0.5 * self.img_h) / self.focal
        uv = np.stack((u, v), -1)
        uvd = np.concatenate((uv, -np.ones([depth.shape[1], 1])), -1)

        pc_cam = np.expand_dims(uvd, 0) * np.expand_dims(depth, 2)
        cam_rot, cam_transl = self.poses[:, :3, :3], self.poses[:, :3, 3]
        pc_world = np.einsum('kij,knj->kni', cam_rot, pc_cam) + np.expand_dims(cam_transl, 1)
        return pc_world


def vis_cameras(poses, focal, img_w, img_h, color=[1., 0., 0.], scale=1.):
    """
    :param poses: (Nv, 4, 4).
    """
    color = np.array(color)
    intrinsic = np.array([[focal, 0., -img_w / 2],
                          [0., -focal, -img_h / 2],
                          [0., 0., -1.]])

    pcds = []
    for pose in poses:
        extrinsic = np.eye(4)
        R, t = pose[:3, :3], pose[:3, 3]
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = - np.dot(R.T, t)

        cam_pcd = o3d.geometry.LineSet()
        cam_pcd = cam_pcd.create_camera_visualization(view_width_px=int(img_w),
                                                      view_height_px=int(img_h),
                                                      intrinsic=intrinsic,
                                                      extrinsic=extrinsic)
        cam_pcd.paint_uniform_color(color)
        cam_pcd.colors[4] = 0.5 * color
        cam_pcd.scale(scale=scale, center=t)
        pcds.append(cam_pcd)
    return pcds

def vis_camera_diff(poses_gt, poses, focal, img_w, img_h, scale):
    """
    :param poses_gt: (Nv, 4, 4).
    :param poses: (Nv, 4, 4).
    """
    n_cam = poses_gt.shape[0]
    color_gt = [0., 1., 0.]
    color = [0., 0., 1.]
    color_diff = [1., 0., 0.]

    pcd_cam_gt = vis_cameras(poses_gt, focal, img_w, img_h,
                             color=color_gt, scale=scale)
    pcd_cam = vis_cameras(poses, focal, img_w, img_h,
                          color=color, scale=scale)

    pcd_camo_gt = o3d.geometry.PointCloud()
    pcd_camo_gt.points = o3d.utility.Vector3dVector(poses_gt[:, :3, 3])
    pcd_camo_gt.colors = o3d.utility.Vector3dVector(np.tile(color_gt, [n_cam, 1]))

    pcd_camo = o3d.geometry.PointCloud()
    pcd_camo.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    pcd_camo.colors = o3d.utility.Vector3dVector(np.tile(color, [n_cam, 1]))

    corr = [(vid, vid) for vid in range(n_cam)]
    pcd_diff = o3d.geometry.LineSet()
    pcd_diff = pcd_diff.create_from_point_cloud_correspondences(pcd_camo_gt, pcd_camo, corr)
    pcd_diff.paint_uniform_color(color_diff)

    pcds = pcd_cam_gt + pcd_cam + [pcd_camo_gt] + [pcd_camo] + [pcd_diff]
    return pcds


def vis_ray(rays_o, rays_d, z=1., color=[1., 0., 0.]):
    """
    :param rays_o: (N, 3).
    :param rays_d: (N, 3).
    :return:
    """
    n_point = rays_o.shape[0]
    corr = [(n, n) for n in range(n_point)]

    start_point = o3d.geometry.PointCloud()
    start_point.points = o3d.utility.Vector3dVector(rays_o)

    end_point = o3d.geometry.PointCloud()
    end_point.points = o3d.utility.Vector3dVector(rays_o + z * rays_d)

    pcd_ray = o3d.geometry.LineSet()
    pcd_ray = pcd_ray.create_from_point_cloud_correspondences(start_point, end_point, corr)
    pcd_ray.paint_uniform_color(color)
    return pcd_ray