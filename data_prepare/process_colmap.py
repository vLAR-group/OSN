import sys
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import yaml
import argparse
import numpy as np
import json
import open3d as o3d

import colmap_utils.colmap_read_model as read_model
from utils.visual_cam_util import vis_cameras


def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    # print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h, w, f])

    imagesfile = os.path.join(realdir, 'images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    names = np.array(names)[perm].tolist()
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    points3dfile = os.path.join(realdir, 'points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # (x, y, z) to (x, -y, -z)
    poses = np.concatenate([c2w_mats[:, :, 0:1], -c2w_mats[:, :, 1:2], -c2w_mats[:, :, 2:3], c2w_mats[:, :, 3:4]], 2)

    return poses, pts3d, perm, hwf, names


def get_colmap_pts3d(pts3d):
    pts_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
    pts_arr = np.array(pts_arr)
    return pts_arr

def get_colmap_pts_err(pts3d):
    pts_err = []
    for k in pts3d:
        pts_err.append(pts3d[k].error)
    pts_err = np.array(pts_err)
    return pts_err


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Load full image names
    image_names_full = sorted(os.listdir(osp.join(args.data_root, 'images_train')))
    n_view = len(image_names_full)

    # Load colmap camera poses
    poses_colmap, points_colmap, visibles_colmap = [], [], []
    for obj_id in range(args.n_object):
        colmap_dir = osp.join(args.data_root, args.preproc_path, 'sparse%d/0' % (obj_id))
        poses, pts3d, perm, hwf, image_names = load_colmap_data(colmap_dir)

        poses_full = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        poses_full = np.tile(poses_full, (n_view, 1, 1))
        visibles = np.zeros((n_view), dtype=np.int32)
        for v, image_name in enumerate(image_names):
            vid = image_names_full.index(image_name)
            poses_full[vid] = poses[v]
            visibles[vid] = 1

        poses_colmap.append(poses_full)
        points_colmap.append(pts3d)
        visibles_colmap.append(visibles)

    poses_colmap = np.stack(poses_colmap, 0)  # (K, Nv, 4, 4)
    img_h, img_w, focal = hwf


    # Save path for processed results
    pose_dir = osp.join(args.data_root, args.preproc_path, 'poses')
    os.makedirs(pose_dir, exist_ok=True)
    depth_dir = osp.join(args.data_root, args.preproc_path, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    # Set scene bound
    near, far = args.near, args.far


    # Process for depth supervision in NeRF optimization
    for obj_id in range(args.n_object):
        poses = poses_colmap[obj_id]
        pts3d = points_colmap[obj_id]
        visibles = visibles_colmap[obj_id]
        points = get_colmap_pts3d(pts3d)

        # Compute points weight according to reprojection error
        points_error = get_colmap_pts_err(pts3d)
        error_mean = points_error.mean()
        points_weight = 2 * np.exp(-(points_error/error_mean)**2)

        # Project 3D point to per-view "uvd"
        images = read_model.read_images_binary(osp.join(args.data_root, args.preproc_path, 'sparse%d/0/images.bin' % (obj_id)))
        uvd_list, vid_list = [], []
        for idx, k in enumerate(pts3d):
            point = points[idx]
            error = points_error[idx]
            weight = points_weight[idx]

            for id_im, pixel_id in zip(pts3d[k].image_ids, pts3d[k].point2D_idxs):
                pose = poses[id_im - 1]
                R, t = pose[:3, :3], pose[:3, 3]

                # # Use reprojected uv location
                # uvd = np.dot(R.T, point) - np.dot(R.T, t)
                # d = - uvd[2]
                # u = focal * uvd[0] / d + 0.5 * img_w
                # v = - focal * uvd[1] / d + 0.5 * img_h

                # Use original uv location
                u, v = images[id_im].xys[pixel_id]
                d = - np.dot(R[:3, 2], point - t)

                uvd = np.array([u, v, d, error, weight])
                uvd_list.append(uvd)
                vid_list.append(id_im - 1)

        uvd_list = np.array(uvd_list, dtype=np.float32)
        vid_list = np.array(vid_list, dtype=np.int32)

        # Drop too far/close points in each view (likely to be outliers) and calculate current scene bounds
        uvd_list_new, vid_list_new = [], []
        close_bounds, inf_bounds = [], []
        for vid in range(n_view):
            if visibles[vid] == 0:
                continue

            uvds, vids = uvd_list[vid_list == vid], vid_list[vid_list == vid]
            depth = uvds[:, 2]

            close_depth, inf_depth = np.percentile(depth, 5), np.percentile(depth, 95)
            close_bounds.append(close_depth)
            inf_bounds.append(inf_depth)

            is_inlier = np.logical_and(depth > close_depth, depth < inf_depth)
            uvds, vids = uvds[is_inlier], vids[is_inlier]
            uvd_list_new.append(uvds)
            vid_list_new.append(vids)
        uvd_list = np.concatenate(uvd_list_new, 0)
        vid_list = np.concatenate(vid_list_new, 0)
        print('Object %d: %d 3D keypoints, %d 2D keypoints'%(obj_id, len(points), len(uvd_list)))

        # Initialize the scale factors for objects: BG fit the scene bound, FG objects in the middle
        close_bound, inf_bound = np.min(close_bounds), np.max(inf_bounds)
        print(close_bound, inf_bound)
        if obj_id == 0:
            scale = 0.95 * far / inf_bound
        else:
            close_scale = 1.05 * near / close_bound
            inf_scale = 0.95 * far / inf_bound
            scale = 0.5 * (close_scale + inf_scale)
            close_scale, inf_scale = close_scale / scale, inf_scale / scale
            print('scale range:', close_scale, inf_scale)

        # Apply the scale factor
        uvd_list[:, 2] *= scale
        poses[:, :3, 3] *= scale
        inf_bound, close_bound = inf_bound * scale, close_bound * scale
        print('Object %d bounds:'%(obj_id), close_bound, inf_bound)

        # Save the results
        pose_path = osp.join(pose_dir, 'pose%d.npy' % (obj_id))
        np.save(pose_path, poses)
        vis_path = osp.join(pose_dir, 'vis%d.npy' % (obj_id))
        np.save(vis_path, visibles)

        depth_path = osp.join(depth_dir, 'depth%d.npy' % (obj_id))
        np.save(depth_path, uvd_list)
        vid_path = osp.join(depth_dir, 'vid%d.npy' % (obj_id))
        np.save(vid_path, vid_list)

        if obj_id > 0:
            scale_path = osp.join(depth_dir, 'scale%d.npy' % (obj_id))
            np.save(scale_path, np.array([close_scale, inf_scale]))


    # Compute oriented bounding boxes for objects
    for obj_id in range(args.n_object):
        pose = np.load(osp.join(pose_dir, 'pose%d.npy' % (obj_id)))
        depth = np.load(osp.join(depth_dir, 'depth%d.npy' % (obj_id)))
        vid = np.load(osp.join(depth_dir, 'vid%d.npy' % (obj_id)))

        # Convert to canonical coordinate system of object
        uvd = depth[:, :3]
        uvd[:, 0] = uvd[:, 2] * (uvd[:, 0] - 0.5 * img_w) / focal
        uvd[:, 1] = -uvd[:, 2] * (uvd[:, 1] - 0.5 * img_h) / focal
        uvd[:, 2] = -uvd[:, 2]
        R, t = pose[vid, :3, :3], pose[vid, :3, 3]
        pts = np.einsum('nij,nj->ni', R, uvd) + t

        # Compute oriented bounding box
        pts_o3d = o3d.geometry.PointCloud()
        pts_o3d.points = o3d.utility.Vector3dVector(pts)
        bbox_o3d = pts_o3d.get_minimal_oriented_bounding_box()

        # Convert to canonical object coordinate system (defined by oriented bounding box)
        R, t = bbox_o3d.R, bbox_o3d.center
        pose_cano = np.eye(4)
        pose_cano[:3, :3], pose_cano[:3, 3] = R.T, - np.dot(R.T, t)
        pose = np.einsum('ij,njk->nik', pose_cano, pose)
        bbox_x, bbox_y, bbox_z = 0.55 * bbox_o3d.extent  # 0.5 means compact, slightly enlarge it
        bbox = np.array([[-bbox_x, bbox_x],
                         [-bbox_y, bbox_y],
                         [-bbox_z, bbox_z]], dtype=np.float32)

        print(bbox)

        # Visualize object & bounding box & camera poses for check
        pts = np.einsum('ij,nj->ni', R.T, pts - t)
        pts_o3d.points = o3d.utility.Vector3dVector(pts)
        color = 0.5 * np.ones((len(pts), 3), dtype=np.float32)
        pts_o3d.colors = o3d.utility.Vector3dVector(color)

        bbox_o3d = o3d.geometry.OrientedBoundingBox()
        bbox_o3d.extent = 2 * np.array([bbox_x, bbox_y, bbox_z], dtype=np.float32)
        bbox_o3d.color = (1, 0, 0)

        cam_o3d = vis_cameras(pose, focal, img_w, img_h, color=[0., 1., 0.], scale=0.25)
        coord_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries(cam_o3d + [pts_o3d, bbox_o3d, coord_o3d])

        # Save the results
        pose_path = osp.join(pose_dir, 'pose%d.npy' % (obj_id))
        np.save(pose_path, pose)
        bbox_path = osp.join(pose_dir, 'bbox%d.npy' % (obj_id))
        np.save(bbox_path, bbox)
        
    
    """
    Measure error between Colmap and GT poses;
    Record GT per-object scales
    """
    # if args.dataset_type == 'blender':
    #     import torch
    #     from datasets.blender import BlenderDataset
    #     from pose import compose_cam_pose
    #     from metric import absolute_traj_error
    #
    #     dataset = BlenderDataset(data_root=args.data_root,
    #                              split='train')
    #     poses_gt = dataset.load_all_poses()
    #     poses_gt = torch.Tensor(poses_gt)
    #     poses_gt = compose_cam_pose(poses_gt[:, 0], poses_gt[:, 1:])  # (Nv, K, 4, 4)
    #     poses_gt = poses_gt.permute(1, 0, 2, 3)  # (K, Nv, 4, 4)
    #
    #     scales_gt = []
    #     for obj_id in range(args.n_object):
    #         poses = np.load(osp.join(pose_dir, 'pose%d.npy'%(obj_id)))
    #         poses = torch.Tensor(poses)
    #         R_error, t_error, sim_s, _, _ = absolute_traj_error(poses, poses_gt[obj_id])
    #         print('Object %d: R_error %.4f, t_error %.4f'%(obj_id, R_error, t_error))
    #         scales_gt.append(sim_s.numpy())
    #
    #     scales_gt = np.array(scales_gt)
    #     scales_gt_path = osp.join(depth_dir, 'scales_gt.npy')
    #     np.save(scales_gt_path, scales_gt)
    #     print('GT scales to align all objects (including BG):', scales_gt)
    #     print('Relative GT scales to align FG objects to BG:', scales_gt[1:] / scales_gt[0])


    """
    Collect meta info for Dynamic Scene dataset
    """
    if args.dataset_type == 'dynamic_scene':
        meta_train = {'focal': focal,
                      'img_h': img_h,
                      'img_w': img_w,
                      'frames': []}
        meta_test = {'focal': focal,
                     'img_h': img_h,
                     'img_w': img_w,
                     'frames': []}

        # Load camera poses
        cam_pose = np.load(osp.join(pose_dir, 'pose0.npy'))
        for v in range(n_view):
            time = v / (n_view - 1)

            frame_data = {'transform_matrix': listify_matrix(cam_pose[v]),
                          'time': time,
                          'time_step': v,
                          'view': v,
                          'file_path': 'train/%04d'%(v)}
            meta_train['frames'].append(frame_data)

            frame_data = {'transform_matrix': listify_matrix(cam_pose[0]),
                          'time': time,
                          'time_step': v,
                          'view': 0,
                          'file_path': 'test/%04d' % (v)}
            meta_test['frames'].append(frame_data)

        with open(osp.join(args.data_root, 'transforms_train.json'), 'w') as fw:
            json.dump(meta_train, fw, indent=4)
        with open(osp.join(args.data_root, 'transforms_test.json'), 'w') as fw:
            json.dump(meta_test, fw, indent=4)

    elif args.dataset_type in ['multimotion', 'iphone', 'kitti']:
        meta_train = {'focal': focal,
                      'img_h': img_h,
                      'img_w': img_w,
                      'frames': []}

        # Load camera poses
        cam_pose = np.load(osp.join(pose_dir, 'pose0.npy'))
        for v in range(n_view):
            time = v / (n_view - 1)

            frame_data = {'transform_matrix': listify_matrix(cam_pose[v]),
                          'time': time,
                          'time_step': v,
                          'view': v,
                          'file_path': 'train/%04d' % (v)}
            meta_train['frames'].append(frame_data)

        with open(osp.join(args.data_root, 'transforms_train.json'), 'w') as fw:
            json.dump(meta_train, fw, indent=4)


    """
    Visualize for check
    """
    # from utils.visual_util import COLOR20, build_colored_pointcloud
    #
    # # Load processed poses & depths
    # poses, depths, vids = [], [], []
    # for obj_id in range(args.n_object):
    #     pose = np.load(osp.join(pose_dir, 'pose%d.npy'%(obj_id)))
    #     poses.append(pose)
    #     depth = np.load(osp.join(depth_dir, 'depth%d.npy' % (obj_id)))
    #     depths.append(depth)
    #     vid = np.load(osp.join(depth_dir, 'vid%d.npy' % (obj_id)))
    #     vids.append(vid)
    #
    # # Visualize the whole scene in each view
    # for v in range(n_view):
    #     pcds = []
    #     for obj_id in range(args.n_object):
    #         uvd, vid = depths[obj_id], vids[obj_id]
    #         uvd = uvd[vid == v, :3]
    #         uvd[:, 0] = uvd[:, 2] * (uvd[:, 0] - 0.5 * img_w) / focal
    #         uvd[:, 1] = -uvd[:, 2] * (uvd[:, 1] - 0.5 * img_h) / focal
    #         uvd[:, 2] = -uvd[:, 2]
    #         color = np.tile(COLOR20[obj_id] / 255, [len(uvd), 1])
    #         pcd = build_colored_pointcloud(uvd, color)
    #         pcds.append(pcd)
    #     # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    #     # pcds.append(coord_frame)
    #     o3d.visualization.draw_geometries(pcds)



    # # Visualize each object in all views
    # for obj_id in range(args.n_object):
    #     points = []
    #     for v in range(n_view):
    #         pose, uvd, vid = poses[obj_id], depths[obj_id], vids[obj_id]
    #         uvd = uvd[vid == v, :3]
    #         uvd[:, 0] = uvd[:, 2] * (uvd[:, 0] - 0.5 * img_w) / focal
    #         uvd[:, 1] = -uvd[:, 2] * (uvd[:, 1] - 0.5 * img_h) / focal
    #         uvd[:, 2] = -uvd[:, 2]
    #         # Camera coordinate to world coordinate
    #         pts = np.einsum('ij,nj->ni', pose[v, :3, :3], uvd) + pose[v, :3, 3]
    #         points.append(pts)
    #     points = np.concatenate(points, 0)
    #     color = np.tile(COLOR20[obj_id] / 255, [len(points), 1])
    #     pose = poses[obj_id]
    #
    #     # # Visualize directly
    #     # pcd_pts = build_colored_pointcloud(points, color)
    #     # pcds = vis_cameras(pose, focal, img_w, img_h, scale=0.5)
    #     # pcds.append(pcd_pts)
    #     # o3d.visualization.draw_geometries(pcds)
    #
    #     # Align to coordinate of GT camera poses then visualize
    #     pose_gt = poses_gt[obj_id]
    #     _, _, sim_s, sim_R, sim_t = absolute_traj_error(torch.Tensor(pose), pose_gt)
    #     sim_s, sim_R, sim_t = sim_s.numpy(), sim_R.numpy(), sim_t.numpy()
    #     sim_T = np.eye(4)
    #     sim_T[:3, :3] = sim_R
    #     sim_T[:3, 3] = sim_t
    #     pose[:, :3, 3] *= sim_s
    #     pose_aligned = np.einsum('ij,njk->nik', sim_T, pose)
    #
    #     pcd_cam_gt = vis_cameras(pose_gt.numpy(), focal, img_w, img_h, scale=0.5, color=[0., 1., 0.])
    #     pcd_cam = vis_cameras(pose_aligned, focal, img_w, img_h, scale=0.5, color=[1., 0., 0.])
    #     points *= sim_s
    #     points = np.einsum('ij,nj->ni', sim_R, points) + sim_t
    #     pcd_pts = build_colored_pointcloud(points, color)
    #     pcds = pcd_cam_gt + pcd_cam + [pcd_pts]
    #     o3d.visualization.draw_geometries(pcds)