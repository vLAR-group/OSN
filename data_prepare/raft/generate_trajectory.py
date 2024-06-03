import argparse
import os
import os.path as osp
import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def sample_coords(img_h, img_w, sample_h, sample_w):
    h_coords = np.linspace(0, img_h - 1, sample_h+1)
    h_coords = 0.5 * (h_coords[1:] + h_coords[:-1])
    w_coords = np.linspace(0, img_w - 1, sample_w+1)
    w_coords = 0.5 * (w_coords[1:] + w_coords[:-1])

    h_coords, w_coords = np.meshgrid(h_coords, w_coords, indexing='ij')
    h_coords, w_coords = h_coords.flatten(), w_coords.flatten()
    coords = np.stack([w_coords, h_coords], axis=1)     # N x (w, h)
    return coords


def extract_feat(uv, feat_map, img_h, img_w):
    """
    :param uv: (N, 2).
    :param feat_map: (H, W, C).
    :return:
        feat: (N, C).
    """
    uv = torch.Tensor(uv).unsqueeze(0).unsqueeze(1)  # (1, 1, N, 2)
    u = 2 * uv[..., 0] / img_w - 1
    v = 2 * uv[..., 1] / img_h - 1
    uv = torch.stack((u, v), -1)
    feat_map = torch.Tensor(feat_map).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    feat = F.grid_sample(feat_map, uv, mode='bilinear', padding_mode='zeros')  # (1, C, 1, N)
    feat = feat.permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # (N, C)
    feat = feat.cpu().numpy()
    return feat


# def find_nonoverlap_points(points, cur_points, radius=5):
#     """
#     :param points: (N1, 2)
#     :param cur_points: (N2, 2)
#     """
#     dist_mat = cdist(points, cur_points)
#     dist_min = np.min(dist_mat, axis=1)
#     nonoverlap_points = points[dist_min > radius]
#     return nonoverlap_points


def find_nonoverlap_points(points, cur_points, radius=5):
    """
    :param points: (N1, 2)
    :param cur_points: (N2, 2)
    """
    points = torch.Tensor(points).cuda().unsqueeze(0)  # (1, N1, 2)
    cur_points = torch.Tensor(cur_points).cuda().unsqueeze(0)  # (1, N2, 2)
    dist_min, _, _ = knn_points(points, cur_points, K=1)  # (1, N1, 1)
    dist_min = dist_min.sqrt()

    points = points[0].cpu().numpy()
    dist_min = dist_min[0, :, 0].cpu().numpy()
    nonoverlap_points = points[dist_min > radius]
    return nonoverlap_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--preproc_path", type=str)
    parser.add_argument("--downsample", type=float)
    args = parser.parse_args()

    data_root = args.data_root
    dataset_name = data_root.split('/')[-1]

    flow_dir = osp.join(data_root, 'flow_raft')
    flow_files = list(sorted(os.listdir(flow_dir)))
    flow_files = [f for f in flow_files if f.endswith('fwd.npz')]

    # Load RAFT estimates
    flows, masks = [], []
    for flow_file in flow_files:
        flow_data = np.load(osp.join(flow_dir, flow_file))
        flows.append(flow_data['flow'])
        masks.append(flow_data['mask'])
    img_h, img_w = flows[0].shape[:2]
    n_view = len(flows) + 1

    # Sample coordinates for trajectories tracking
    downsample = args.downsample
    sample_h, sample_w = int(img_h * np.sqrt(downsample)), int(img_w * np.sqrt(downsample))
    coords = sample_coords(img_h, img_w, sample_h, sample_w)
    radius = int(np.sqrt(1/downsample))

    save_dir = osp.join(data_root, args.preproc_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Add first-frame sampled points into trajectories
    trajectories = np.zeros([coords.shape[0], n_view, 2], dtype=np.float32)     # (N, V, 2)
    trajectories[:, 0] = coords
    visibilities = np.zeros([coords.shape[0], n_view], dtype=np.int32)      # (N, V)
    visibilities[:, 0] = 1
    
    # Track along frames
    cur_coords = coords.copy()
    cur_coord_ids = np.arange(coords.shape[0], dtype=np.int32)
    for vid in range(1, n_view):
        flow, mask = flows[vid-1], masks[vid-1].astype(np.float32)
        flow = extract_feat(cur_coords, flow, img_h, img_w)
        mask = extract_feat(cur_coords, np.expand_dims(mask, 2), img_h, img_w).squeeze(1)

        mask = (mask > 0.95)
        cur_coords, cur_coord_ids, flow = cur_coords[mask], cur_coord_ids[mask], flow[mask]

        next_coords = cur_coords + flow
        trajectories[cur_coord_ids, vid] = next_coords
        visibilities[cur_coord_ids, vid] = 1

        # Update more tracking points
        new_coords = find_nonoverlap_points(coords, next_coords, radius=radius)
        new_coord_ids = np.arange(new_coords.shape[0], dtype=np.int32) + trajectories.shape[0]
        new_trajectories = np.zeros([new_coords.shape[0], n_view, 2], dtype=np.float32)
        new_trajectories[:, vid] = new_coords
        new_visibilities = np.zeros([new_coords.shape[0], n_view], dtype=np.int32)
        new_visibilities[:, vid] = 1

        trajectories = np.concatenate([trajectories, new_trajectories], axis=0)
        visibilities = np.concatenate([visibilities, new_visibilities], axis=0)
        cur_coords = np.concatenate([next_coords, new_coords], axis=0)
        cur_coord_ids = np.concatenate([cur_coord_ids, new_coord_ids], axis=0)

    # Remove trajectories with too few frames
    vis_nframe = visibilities.sum(1)
    # valid = (vis_nframe > 3)
    valid = (vis_nframe > 1)
    trajectories, visibilities = trajectories[valid], visibilities[valid]


    """
    Assign trajectories to objects according to predicted segmentation
    """
    segm_dir = osp.join(data_root, 'segm_sam')
    print('Load SAM segmentation from %s' % (segm_dir))

    segm_files = sorted(os.listdir(segm_dir))
    segms = []
    for vid in range(n_view):
        segm = np.load(osp.join(segm_dir, segm_files[vid]))
        segm = segm.astype(np.float32)
        coords = trajectories[:, vid]
        segm = extract_feat(coords, np.expand_dims(segm, 2), img_h, img_w).squeeze(1)
        segm = np.round(segm).astype(np.int32)
        segms.append(segm)
    segms = np.stack(segms, axis=1)     # (N, V, C)

    # Remove trajectories with inconsistent segmentation along frames
    inlier_ids, segms_new = [], []
    for n in range(trajectories.shape[0]):
        segm, vis = segms[n], visibilities[n]
        segm = segm[vis > 0]
        if np.all(segm == segm[0]):
            inlier_ids.append(n)
            segms_new.append(segm[0])
    inlier_ids = np.array(inlier_ids, dtype=np.int32)
    segms = np.stack(segms_new, axis=0, dtype=np.int32)

    trajectories, visibilities = trajectories[inlier_ids], visibilities[inlier_ids]

    print('Number of trajectories: %d' % (trajectories.shape[0]))
    # Save the results
    np.save(osp.join(save_dir, 'trajectories.npy'), trajectories)
    np.save(osp.join(save_dir, 'visibilities.npy'), visibilities)
    np.save(osp.join(save_dir, 'segm_preds.npy'), segms)