import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import json
import numpy as np
import open3d as o3d

import torch

from metric import absolute_traj_error
from utils.visual_cam_util import vis_cameras


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if __name__ == "__main__":
    data_root = '/media/SSD/ziyang/Datasets_NeRF/KITTI/data/0007'
    data_root_stereo = data_root + '_stereo'
    preproc_path = 'sam_raft'

    with open(osp.join(data_root, 'transforms_train.json'), 'r') as f:
        meta = json.load(f)
    img_h, img_w, focal = float(meta['img_h']), float(meta['img_w']), float(meta['focal'])

    # Load train poses
    pose_dir = osp.join(data_root, preproc_path, 'poses')
    poses = np.load(osp.join(pose_dir, 'pose0.npy'))

    # Load test poses and 3D points
    pose_dir = osp.join(data_root_stereo, preproc_path, 'poses')
    poses_stereo = np.load(osp.join(pose_dir, 'pose0.npy'))

    # Split stereo poses to left and right camera
    poses_left = poses_stereo[::2]
    poses_right = poses_stereo[1::2]

    # Align left camera to training poses
    R_error, t_error, s, R, t = absolute_traj_error(torch.from_numpy(poses_left).float(), torch.from_numpy(poses).float())
    print('R_error:', R_error, 't_error:', t_error)
    s, R, t = s.cpu().numpy(), R.cpu().numpy(), t.cpu().numpy()

    # Use the alignment to align right camera
    poses_test = poses_right.copy()
    poses_test[:, :3, :3] = np.einsum('ij,njk->nik', R, poses_test[:, :3, :3])
    poses_test[:, :3, 3] = s * np.einsum('ij,nj->ni', R, poses_test[:, :3, 3]) + t

    # Check by visualization
    cam_o3d_train = vis_cameras(poses, focal, img_w, img_h, color=[0., 1., 0.], scale=0.25)
    cam_o3d_test = vis_cameras(poses_test, focal, img_w, img_h, color=[1., 0., 0.], scale=0.25)
    o3d.visualization.draw_geometries(cam_o3d_train + cam_o3d_test)

    # Write testing meta
    meta_test = {'focal': focal,
                 'img_h': img_h,
                 'img_w': img_w,
                 'frames': []}

    # Load camera poses
    n_view = poses_test.shape[0]
    for v in range(n_view):
        time = v / (n_view - 1)

        frame_data = {'transform_matrix': listify_matrix(poses_test[v]),
                      'time': time,
                      'time_step': v,
                      'view': v,
                      'file_path': 'test/%04d' % (v)}
        meta_test['frames'].append(frame_data)

    with open(osp.join(data_root, 'transforms_test.json'), 'w') as fw:
        json.dump(meta_test, fw, indent=4)


print('Good')