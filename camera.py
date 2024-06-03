import numpy as np
import torch


def convert_rays_to_ndc(rays_o, rays_d, img_h, img_w, focal, near_plane=1.):
    """
    :param rays_o: (Nr, 3) torch.Tensor.
    :param rays_d: (Nr, 3) torch.Tensor.
    :return:
        rays_o: (Nr, 3) torch.Tensor.
        rays_d: (Nr, 3) torch.Tensor.
    """
    # Shift "o" to the ray's intersection with the near plane
    t = -(near_plane + rays_o[:, 2]) / rays_d[:, 2]
    rays_o = rays_o + t.unsqueeze(1) * rays_d

    # Convert to NDC
    o0 = - (2 * focal / img_w) * (rays_o[:, 0] / rays_o[:, 2])
    o1 = - (2 * focal / img_h) * (rays_o[:, 1] / rays_o[:, 2])
    o2 = 1. + 2. * near_plane / rays_o[:, 2]

    d0 = - (2 * focal / img_w) * (rays_d[:, 0] / rays_d[:, 2] - rays_o[:, 0] / rays_o[:, 2])
    d1 = - (2 * focal / img_h) * (rays_d[:, 1] / rays_d[:, 2] - rays_o[:, 1] / rays_o[:, 2])
    d2 = - 2. * near_plane / rays_o[:, 2]

    rays_o = torch.stack([o0, o1, o2], 1)
    rays_d = torch.stack([d0, d1, d2], 1)

    return rays_o, rays_d


def restore_ndc_points(points, img_h, img_w, focal, near_plane=1.):
    """
    :param points: (N, 3) torch.Tensor.
    :return:
        points: (N, 3) torch.Tensor.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    p_z = (2 * near_plane) / (z - 1)
    p_x = - x * p_z * img_w / (2 * focal)
    p_y = - y * p_z * img_h / (2 * focal)
    points = torch.stack([p_x, p_y, p_z], 1)
    return points


class Rays:
    def __init__(self,
                 rays_o,
                 rays_d,
                 viewdirs=None,
                 n_sample_point=64,
                 n_sample_point_fine=128,
                 near=2.,
                 far=6.,
                 perturb=False):
        """
        :param rays_o: (Nr, 3) torch.Tensor.
        :param rays_d: (Nr, 3) torch.Tensor.
        """
        self.rays_o = rays_o
        self.rays_d = rays_d
        if viewdirs is None:
            viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
        self.viewdirs = viewdirs

        self.n_sample_point = n_sample_point
        self.n_sample_point_fine = n_sample_point_fine
        self.near = near
        self.far = far
        self.perturb = perturb

    def sample_points(self):
        """
        :return:
            points: (Nr, Np, 3) torch.Tensor.
            viewdirs: (Nr, Np, 3) torch.Tensor.
            z_vals: (Nr, Np) torch.Tensor.
        """
        t_vals = torch.linspace(0., 1., steps=self.n_sample_point)
        z_vals = self.near * (1. - t_vals) + self.far * t_vals

        n_ray = self.rays_o.shape[0]
        z_vals = z_vals.expand([n_ray, self.n_sample_point])

        if self.perturb:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], 1)
            lower = torch.cat([z_vals[:, :1], mids], 1)
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.unsqueeze(2)
        rays_o = self.rays_o.unsqueeze(1).expand([n_ray, self.n_sample_point, 3])
        rays_d = self.rays_d.unsqueeze(1).expand([n_ray, self.n_sample_point, 3])
        points = rays_o + z_vals * rays_d

        viewdirs = self.viewdirs.unsqueeze(1).expand([n_ray, self.n_sample_point, 3])

        return points, viewdirs, z_vals.squeeze(2)

    def sample_points_fine(self, z_vals, weights):
        """
        :param z_vals: (Nr, Np) torch.Tensor.
        :param weights: (Nr, Np) torch.Tensor.
        :return:
            points: (Nr, Np, 3) torch.Tensor.
            viewdirs: (Nr, Np, 3) torch.Tensor.
            z_vals: (Nr, Np) torch.Tensor.
        """
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])       # (Nr, Np - 1)

        weights = weights[:, 1:-1] + 1e-5       # (Nr, Np - 2)
        pdf = weights / torch.sum(weights, 1, keepdim=True)
        cdf = torch.cumsum(pdf, 1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], 1)       # (Nr, Np - 1)
        n_ray = cdf.shape[0]

        if self.perturb:
            u_vals = torch.rand([n_ray, self.n_sample_point_fine])
        else:
            u_vals = torch.linspace(0., 1., steps=self.n_sample_point_fine)
            u_vals = u_vals.expand([n_ray, self.n_sample_point_fine])

        # Inverse transform sampling
        inds = torch.searchsorted(cdf, u_vals, right=True)
        below = (inds - 1).clamp(min=0.)        # (Nr, Np_fine)
        above = inds.clamp(max=(cdf.shape[1]-1))      # (Nr, Np_fine)
        inds_g = torch.stack([below, above], 2)     # (Nr, Np_fine, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[1]]        # (Nr, Np_fine, Np - 1)
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)     # (Nr, Np_fine, 2)
        mids_g = torch.gather(mids.unsqueeze(1).expand(matched_shape), 2, inds_g)       # (Nr, Np_fine, 2)

        denom = (cdf_g[:, :, 1] - cdf_g[:, :, 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_vals = (u_vals - cdf_g[:, :, 0]) / denom
        z_vals_new = (1 - t_vals) * mids_g[:, :, 0] + t_vals * mids_g[:, :, 1]

        # Merge coarse and fine sampled points
        z_vals = torch.cat([z_vals, z_vals_new], 1)
        z_vals, _ = torch.sort(z_vals, 1)       # (Nr, Np + Np_fine)

        z_vals = z_vals.unsqueeze(2)
        rays_o = self.rays_o.unsqueeze(1).expand([n_ray, z_vals.shape[1], 3])
        rays_d = self.rays_d.unsqueeze(1).expand([n_ray, z_vals.shape[1], 3])
        points = rays_o + z_vals * rays_d

        viewdirs = self.viewdirs.unsqueeze(1).expand([n_ray, z_vals.shape[1], 3])

        return points, viewdirs, z_vals.squeeze(2)


class Camera:
    def __init__(self, img_h, img_w, focal, pose):
        """
        :param pose: (4, 4) torch.Tensor.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.focal = focal
        self.pose = pose

    def get_rays_np(self):
        """
        :return:
            rays_o: (H, W, 3).
            rays_d: (H, W, 3).
        """
        u, v = np.meshgrid(np.arange(self.img_w, dtype=np.float32), np.arange(self.img_h, dtype=np.float32),
                           indexing='xy')

        u = (u - 0.5 * self.img_w) / self.focal
        v = - (v - 0.5 * self.img_h) / self.focal
        uvd = np.stack((u, v, -np.ones_like(u)), -1)
        rays_d = np.sum(uvd[..., np.newaxis, :] * self.pose[:3,:3], -1)
        rays_o = np.broadcast_to(self.pose[:3, 3], rays_d.shape)

        return rays_o, rays_d

    def get_rays(self):
        """
        :return:
            rays_o: (H, W, 3) torch.Tensor.
            rays_d: (H, W, 3) torch.Tensor.
        """
        u, v = torch.meshgrid(torch.arange(self.img_w), torch.arange(self.img_h))
        u, v = u.t(), v.t()  # 'ij' indexing to 'xy' indexing

        u = (u - 0.5 * self.img_w) / self.focal
        v = - (v - 0.5 * self.img_h) / self.focal
        uvd = torch.stack((u, v, -torch.ones_like(u)), -1)
        # rays_d = torch.einsum('ij,mnj->mni', self.pose[:3, :3], uvd)
        rays_d = torch.sum(uvd[..., np.newaxis, :] * self.pose[:3,:3], -1)
        rays_o = self.pose[:3, 3].expand(rays_d.shape)

        return rays_o, rays_d

    def sample_rays(self, n_sample_ray=4096, precrop_frac=1.0):
        """
        :return:
            rays_o: (Nr, 3) torch.Tensor.
            rays_d: (Nr, 3) torch.Tensor.
            select_coords: (Nr, 2) torch.Tensor.
        """
        # Get rays for all pixels
        rays_o, rays_d = self.get_rays()

        # Sample from the rays
        if precrop_frac < 1.0:
            d_h = int(self.img_h // 2 * precrop_frac)
            d_w = int(self.img_w // 2 * precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.img_h // 2 - d_h, self.img_h // 2 + d_h - 1, 2 * d_h),
                    torch.linspace(self.img_w // 2 - d_w, self.img_w // 2 + d_w - 1, 2 * d_w)
                ), -1)
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.img_h - 1, int(self.img_h)),
                    torch.linspace(0, self.img_w - 1, int(self.img_w))
                ), -1)
        coords = torch.reshape(coords, [-1, 2])     # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=n_sample_ray, replace=False)       # (N,)
        select_coords = coords[select_inds].long()      # (N, 2)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]

        return rays_o, rays_d, select_coords


class BatchCameras:
    def __init__(self, img_h, img_w, focal, poses):
        """
        :param poses: (K, 4, 4) torch.Tensor.
        """
        self.img_h = img_h
        self.img_w = img_w
        self.focal = focal
        self.poses = poses

    def get_rays(self):
        """
        :return:
            rays_o: (K, H, W, 3) torch.Tensor.
            rays_d: (K, H, W, 3) torch.Tensor.
        """
        u, v = torch.meshgrid(torch.arange(self.img_w), torch.arange(self.img_h))
        u, v = u.t(), v.t()  # 'ij' indexing to 'xy' indexing

        u = (u - 0.5 * self.img_w) / self.focal
        v = - (v - 0.5 * self.img_h) / self.focal
        uvd = torch.stack((u, v, -torch.ones_like(u)), -1)
        rays_d = torch.einsum('kij,mnj->kmni', self.poses[:, :3, :3], uvd)
        rays_o = self.poses[:, :3, 3].unsqueeze(1).unsqueeze(2).expand(rays_d.shape)

        return rays_o, rays_d

    def sample_rays(self, n_sample_ray=4096, precrop_frac=1.0):
        """
        :return:
            rays_o: (K, Nr, 3) torch.Tensor.
            rays_d: (K, Nr, 3) torch.Tensor.
            select_coords: (K, Nr, 2) torch.Tensor.
        """
        # Get rays for all pixels
        rays_o, rays_d = self.get_rays()

        # Sample from the rays
        if precrop_frac < 1.0:
            d_h = int(self.img_h // 2 * precrop_frac)
            d_w = int(self.img_w // 2 * precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.img_h // 2 - d_h, self.img_h // 2 + d_h - 1, 2 * d_h),
                    torch.linspace(self.img_w // 2 - d_w, self.img_w // 2 + d_w - 1, 2 * d_w)
                ), -1)
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.img_h - 1, int(self.img_h)),
                    torch.linspace(0, self.img_w - 1, int(self.img_w))
                ), -1)
        coords = torch.reshape(coords, [-1, 2])     # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=n_sample_ray, replace=False)       # (N,)
        select_coords = coords[select_inds].long()      # (N, 2)
        rays_d = rays_d[:, select_coords[:, 0], select_coords[:, 1]]
        rays_o = rays_o[:, select_coords[:, 0], select_coords[:, 1]]

        return rays_o, rays_d, select_coords

    def sample_rays_with_mask(self, mask, n_sample_ray=4096, precrop_frac=1.0):
        """
        :param mask: (K, H, W) torch.Tensor.
        :return:
            rays_o: (Nr, 3) torch.Tensor.
            rays_d: (Nr, 3) torch.Tensor.
            select_coords: (Nr, 3) torch.Tensor.
        """
        # Get rays for all pixels
        rays_o, rays_d = self.get_rays()
        mask = mask.reshape(-1)
        n_sample_ray = min(int(mask.float().sum()), n_sample_ray)

        # Sample from the rays
        n_view = self.poses.shape[0]
        if precrop_frac < 1.0:
            d_h = int(self.img_h // 2 * precrop_frac)
            d_w = int(self.img_w // 2 * precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, n_view - 1, n_view),
                    torch.linspace(self.img_h // 2 - d_h, self.img_h // 2 + d_h - 1, 2 * d_h),
                    torch.linspace(self.img_w // 2 - d_w, self.img_w // 2 + d_w - 1, 2 * d_w)
                ), -1)
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, n_view - 1, n_view),
                    torch.linspace(0, self.img_h - 1, int(self.img_h)),
                    torch.linspace(0, self.img_w - 1, int(self.img_w))
                ), -1)
        coords = torch.reshape(coords, [-1, 3])  # (K * H * W, 3)
        coords = coords[mask]
        select_inds = np.random.choice(coords.shape[0], size=n_sample_ray, replace=False)  # (N,)
        select_coords = coords[select_inds].long()  # (N, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]

        return rays_o, rays_d, select_coords

    def sample_rays_from_coords(self, coords, n_sample_ray=4096):
        """
        :param coords: (N, 2) torch.Tensor.
        :return:
            rays_o: (Nr, 3) torch.Tensor.
            rays_d: (Nr, 3) torch.Tensor.
            select_coords: (Nr) torch.Tensor.
        """
        u, v = coords[:, 0], coords[:, 1]
        u = (u - 0.5 * self.img_w) / self.focal
        v = - (v - 0.5 * self.img_h) / self.focal
        uvd = torch.stack((u, v, -torch.ones_like(u)), -1)
        rays_d = torch.einsum('ij,nj->ni', self.poses[0, :3, :3], uvd)
        rays_o = self.poses[0, :3, 3].unsqueeze(0).expand(rays_d.shape)

        n_sample_ray = min(rays_o.shape[0], n_sample_ray)
        select_inds = np.random.choice(rays_o.shape[0], size=n_sample_ray, replace=False)  # (Nr,)
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        return rays_o, rays_d, select_inds