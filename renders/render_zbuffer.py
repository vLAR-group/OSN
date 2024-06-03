import torch


def zbuffer_render(rgb_map_mb, depth_map_mb, acc_map_mb, scales, soft=False, zbuffer_beta=0.1, render_segm_only=True):
    """
    :param rgb_map_mb: (K, Nr, 3) torch.Tensor.
    :param depth_map_mb: (K, Nr) torch.Tensor.
    :param acc_map_mb: (K, Nr) torch.Tensor.
    :param scales: (Ns, K) torch.Tensor.
    :return:
        rgb_map: (Ns, Nr, 3) torch.Tensor.
        depth_map: (Ns, Nr) torch.Tensor.
        segm_map: (Ns, Nr, K) torch.Tensor.
    """
    # Scale the depth maps
    depth_map_mb = depth_map_mb * scales.unsqueeze(2)  # (Ns, K, Nr)
    # Render the segmentation map according to depth buffer
    segm_map = acc_map_mb * torch.exp(- zbuffer_beta * depth_map_mb)  # (Ns, K, Nr)
    segm_map = segm_map.permute(0, 2, 1)  # (Ns, Nr, K)
    # Convert segm map to one-hot
    if soft:
        segm_map = segm_map / segm_map.sum(dim=-1, keepdim=True).clamp(min=1e-10)  # (Ns, Nr, K)
    else:
        n_object = segm_map.shape[-1]
        segm_pred = torch.argmax(segm_map, dim=-1)  # (Ns, Nr)
        segm_map = torch.eye(n_object)[segm_pred]  # (Ns, Nr, K)

    if render_segm_only:
        return segm_map

    else:
        # Blend color & depth maps according to the segmentation map
        segm_map = segm_map.permute(0, 2, 1)  # (Ns, K, Nr)
        rgb_map = torch.sum(segm_map.unsqueeze(3) * rgb_map_mb.unsqueeze(0), 1)  # (Ns, Nr, 3)
        depth_map = torch.sum(segm_map * depth_map_mb, 1)  # (Ns, Nr)
        segm_map = segm_map.permute(0, 2, 1)  # (Ns, Nr, K)
        return rgb_map, depth_map, segm_map