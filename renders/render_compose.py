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
        rgb_map: (Nr, 3) torch.Tensor.
        depth_map: (Nr,) torch.Tensor.
        acc_map: (Nr,) torch.Tensor.
        segm_map: (Nr, K) torch.Tensor.
        weights: (Nr, Np) torch.Tensor.
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([0.]).expand(dists[:, :1].shape)], 1)
    dists = dists * rays_d.norm(dim=1, keepdim=True)    # (Nr, Np)

    # Composite density
    alpha_mb = 1. - torch.exp(- density_mb * dists.unsqueeze(0))      # (K, Nr, Np)
    density = density_mb.sum(0)     # (Nr, Np)
    # Composite RGB
    rgb_mb = rgb_mb * density_mb.unsqueeze(-1)        # (K, Nr, Np, 3)
    rgb = rgb_mb.sum(0) / density.clamp(min=1e-10).unsqueeze(-1)    # (Nr, Np, 3)

    # Compute transparency
    alpha = 1. - torch.exp(- density * dists)       # (Nr, Np)
    transparency = torch.cumprod(torch.cat([torch.ones(alpha[:, :1].shape), 1. - alpha + 1e-10], 1), 1)[:, :-1]     # (Nr, Np)

    # Render segmentation
    # mask = alpha_mb / alpha.unsqueeze(0).clamp(min=1e-10)       # (K, Nr, Np)
    # mask = mask.permute(1, 2, 0)     # (Nr, Np, K)
    mask = alpha_mb.permute(1, 2, 0)        # (Nr, Np, K)
    segm_map = torch.sum(transparency.unsqueeze(2) * mask, 1)      # (Nr, K)

    # Render RGB and depth
    weights = alpha * transparency      # (Nr, Np)
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, 1)      # (Nr, 3)
    depth_map = torch.sum(weights * z_vals, 1)      # (Nr,)
    acc_map = torch.sum(weights, 1)     # (Nr,)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    # Compute entropy
    occup = mask.sum(-1, keepdims=True).clamp(1e-8)
    mask = mask / occup
    entropy = - (mask * torch.log(mask.clamp(1e-8))).sum(dim=-1, keepdims=True)
    entropy = (occup * entropy).sum() / occup.sum()

    return rgb_map, depth_map, acc_map, segm_map, weights, entropy


def nerf_render(rays,
                scales,
                models,
                bboxes=None,
                poses_mb=None,
                visibles=None,
                white_bkgd=False):
    """
    :param poses_mb: (K, Nr, 4, 4) torch.Tensor.
    :param visibles: (K, Nr) torch.Tensor.
    :param segm: (Nr,) torch.Tensor.
    """
    # Sample points on rays
    points, viewdirs, z_vals = rays.sample_points()
    n_ray, n_point = points.shape[:2]
    n_object = len(models)

    rgb_mb, density_mb = [], []
    # Query each point for all objects
    for k in range(n_object):
        poses, model, visible = poses_mb[k], models[k], visibles[k]
        # Scale and transform the points
        points_k = torch.einsum('vij,vnj->vni', poses[:, :3, :3], points) / scales[k] + poses[:, :3, 3].unsqueeze(1)
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
    rgb_map, depth_map, acc_map, segm_map, weights, entropy = volume_render(rgb_mb,
                                                                            density_mb,
                                                                            z_vals,
                                                                            rays_d,
                                                                            white_bkgd)
    depth_map = depth_map + (1. - acc_map) * rays.far
    # Collect_results
    ret_dict = {'rgb_map': rgb_map,
                'depth_map': depth_map,
                'acc_map': acc_map,
                'segm_map': segm_map,
                'entropy': entropy}

    return ret_dict