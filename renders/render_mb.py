import torch


def volume_render(rgb_mb,
                  density_mb,
                  z_vals,
                  rays_d,
                  white_bkgd=False):
    """
    :param rgb_mb: (K, Nr, Np, 3) torch.Tensor.
    :param density_mb: (K, Nr, Np) torch.Tensor.
    :param z_vals: (Nr, Np) torch.Tensor.
    :param rays_d: (Nr, 3) torch.Tensor.
    :return:
        rgb_map_mb: (K, Nr, 3) torch.Tensor.
        depth_map_mb: (K, Nr) torch.Tensor.
        acc_map_mb: (K, Nr) torch.Tensor.
        weights_mb: (k, Nr, Np) torch.Tensor.
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([0.]).expand(dists[:, :1].shape)], 1)
    dists = dists * rays_d.norm(dim=1, keepdim=True)    # (Nr, Np)

    # Render for each object on all pixels
    alpha_mb = 1. - torch.exp(- density_mb * dists.unsqueeze(0))      # (K, Nr, Np)
    transparency_mb = torch.cumprod(torch.cat([torch.ones(alpha_mb[:, :, :1].shape), 1. - alpha_mb + 1e-10], 2), 2)[:, :, :-1]     # (K, Nr, Np)
    weights_mb = alpha_mb * transparency_mb     # (K, Nr, Np)

    rgb_map_mb = torch.sum(weights_mb.unsqueeze(3) * rgb_mb, 2)     # (K, Nr, 3)
    depth_map_mb = torch.sum(weights_mb * z_vals.unsqueeze(0), 2)      # (K, Nr)
    acc_map_mb = torch.sum(weights_mb, 2)  # (K, Nr)

    if white_bkgd:
        rgb_map_mb = rgb_map_mb + (1. - acc_map_mb.unsqueeze(2))

    # Compute entropy
    entropy = torch.Tensor([0.])

    return rgb_map_mb, depth_map_mb, acc_map_mb, weights_mb, entropy


def nerf_render(rays,
                models,
                bboxes=None,
                poses_mb=None,
                visibles=None,
                white_bkgd=False):
    """
    :param poses_mb: (K, Nr, 4, 4) torch.Tensor.
    :param visibles: (K, Nr) torch.Tensor.
    """
    # Sample points on rays
    points, viewdirs, z_vals = rays.sample_points()
    n_ray, n_point = points.shape[:2]
    n_object = len(models)

    rgb_mb, density_mb = [], []
    # Query each point for all objects
    for k in range(n_object):
        poses, model, visible = poses_mb[k], models[k], visibles[k]
        # Transform the points
        points_k = torch.einsum('vij,vnj->vni', poses[:, :3, :3], points) + poses[:, :3, 3].unsqueeze(1)
        viewdirs_k = torch.einsum('vij,vnj->vni', poses[:, :3, :3], viewdirs)
        points_k, viewdirs_k = points_k.reshape(-1, 3), viewdirs_k.reshape(-1, 3)

        # Initialize zero density and color
        rgb, density = torch.zeros(points_k.shape[0], 3), torch.zeros(points_k.shape[0])

        # Exclude rays where object are not visible in the view
        visible = visible.unsqueeze(-1).expand(-1, n_point).reshape(-1)
        visible = (visible > 0)

        # Find points inside the bounding box
        in_bbox_x = torch.logical_and(points_k[:, 0] > bboxes[k][0, 0], points_k[:, 0] < bboxes[k][0, 1])
        in_bbox_y = torch.logical_and(points_k[:, 1] > bboxes[k][1, 0], points_k[:, 1] < bboxes[k][1, 1])
        in_bbox_z = torch.logical_and(points_k[:, 2] > bboxes[k][2, 0], points_k[:, 2] < bboxes[k][2, 1])
        in_bbox = torch.logical_and(in_bbox_x, torch.logical_and(in_bbox_y, in_bbox_z))

        in_bbox = torch.logical_and(in_bbox, visible)
        points_k, viewdirs_k = points_k[in_bbox], viewdirs_k[in_bbox]

        # Query
        if points_k.shape[0] > 0:
            rgb_in_bbox, density_in_bbox = model(points_k, viewdirs_k)
            rgb[in_bbox], density[in_bbox] = rgb_in_bbox, density_in_bbox

        rgb, density = rgb.reshape(n_ray, n_point, 3), density.reshape(n_ray, n_point)

        rgb_mb.append(rgb)
        density_mb.append(density)

    rgb_mb, density_mb = torch.stack(rgb_mb, 0), torch.stack(density_mb, 0)     # (K, Nr, Np, 3), (K, Nr, Np)

    # Differentiable volume render
    z_vals = z_vals
    rays_d = rays.rays_d
    rgb_map_mb, depth_map_mb, acc_map_mb, weights_mb, entropy = volume_render(rgb_mb,
                                                                              density_mb,
                                                                              z_vals,
                                                                              rays_d,
                                                                              white_bkgd)
    depth_map_mb = depth_map_mb + (1. - acc_map_mb) * rays.far
    # Collect_results
    ret_dict = {'rgb_map_mb': rgb_map_mb,
                'depth_map_mb': depth_map_mb,
                'acc_map_mb': acc_map_mb,
                'entropy': entropy}

    return ret_dict