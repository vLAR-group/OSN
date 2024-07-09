import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import imageio.v3 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorf.tensoRF import TensorVMSplit
from tensorf.utils import N_to_reso, cal_n_samples
from pose import LiePose, decompose_cam_pose, compose_cam_pose
from camera import Camera, Rays
from renders.render_compose import nerf_render as nerf_render_compose
from renders.render_composetest import nerf_render as nerf_render_composetest
from utils.visual_util import build_segm_vis


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--checkpoint', type=int, default=200000, help='Checkpoint (iteration) to load')
    parser.add_argument('--n_sample_scale_test', type=int, default=0, help='Number of object scales sampled in prior-free sampling')
    parser.add_argument('--scale_id', type=int, default=0, help='Index of sampled scale in prior-free sampling or scene generation')
    parser.add_argument('--render_test', action='store_true', help='Render test views (else render training views)')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the data
    if args.dataset_type == 'indoor':
        from datasets.blender import BlenderDataset
        train_set = BlenderDataset(data_root=args.data_root,
                                   split='train',
                                   flow_window_size=args.flow_window_size,
                                   white_bkgd=args.white_bkgd)
        test_set = BlenderDataset(data_root=args.data_root,
                                  split='test',
                                  flow_window_size=args.flow_window_size,
                                  white_bkgd=args.white_bkgd)
    elif args.dataset_type in ['dynamic_scene', 'multimotion', 'iphone', 'kitti']:
        from datasets.realworld import RealWorldDataset
        train_set = RealWorldDataset(data_root=args.data_root,
                                     split='train')
        test_set = RealWorldDataset(data_root=args.data_root,
                                    split='test')
    else:
        raise ValueError('Not implemented!')
    n_view = train_set.n_sample
    n_test_sample = test_set.n_sample
    print('%d testing samples'%(n_test_sample))

    img_h, img_w, focal = train_set.img_h, train_set.img_w, train_set.focal
    imgs_train = test_set.load_all_images()
    imgs_train = torch.Tensor(imgs_train)

    # iPhone dataset has different focal length for training and testing
    if args.dataset_type == 'iphone' and args.render_test:
        focal = test_set.focal
    print(focal)

    # Define bounds
    near, far = args.near, args.far
    near_far = torch.Tensor([near, far])
    print('near: %.3f, far: %.3f' % (near, far))


    exp_base = args.exp_base

    # Load Bounding box for each object
    bboxes = []
    for k in range(args.n_object):
        bbox = np.load(osp.join(args.data_root, args.preproc_path, 'poses/bbox%d.npy' % (k)))
        bbox = torch.Tensor(bbox)
        bboxes.append(bbox)

    # TesnsoRF coarse-to-fine upsampling: linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(
        np.log(args.tensorf['N_voxel_init']), np.log(args.tensorf['N_voxel_final']), len(args.tensorf['upsamp_list']) + 1
    ))).long()).tolist()
    # Find the resolution for to-be-loaded checkpoint
    upsamp_list = args.tensorf['upsamp_list']
    upsamp_list = [0] + upsamp_list
    upsamp_list = np.array(upsamp_list, dtype=np.int32)
    upsamp_list = upsamp_list[args.checkpoint > upsamp_list]
    reso_idx = len(upsamp_list) - 1
    n_voxels = N_voxel_list[reso_idx]

    # Create the network (coarse) and load trained model weights
    models = []
    n_sample_points = []
    for k in range(args.n_object):
        aabb = bboxes[k].transpose(0, 1)
        reso_cur = N_to_reso(n_voxels, aabb)
        model = TensorVMSplit(aabb=aabb,
                              gridSize=reso_cur,
                              device=aabb.device,
                              density_n_comp=args.tensorf['n_lamb_sigma'],
                              appearance_n_comp=args.tensorf['n_lamb_sh'],
                              app_dim=args.tensorf['data_dim_color'],
                              near_far=near_far,
                              shadingMode=args.tensorf['shadingMode'],
                              alphaMask_thres=args.tensorf['alpha_mask_thre'],
                              density_shift=args.tensorf['density_shift'],
                              distance_scale=args.tensorf['distance_scale'],
                              pos_pe=args.tensorf['pos_pe'],
                              view_pe=args.tensorf['view_pe'],
                              fea_pe=args.tensorf['fea_pe'],
                              featureC=args.tensorf['featureC'],
                              fea2denseAct=args.tensorf['fea2denseAct'])
        weight_path = osp.join(exp_base, 'model_%06d_%02d.pth.tar' % (args.checkpoint, k))
        model.load_state_dict(torch.load(weight_path))
        models.append(model)
        n_sample_points.append(cal_n_samples(reso_cur, step_ratio=0.5))

    if args.n_sample_point_adjust:
        n_sample_point = max(n_sample_points)
    else:
        n_sample_point = args.n_sample_point
    print('Set n_sample_point (per ray) to %d'%(n_sample_point))

    # Create the pose parameters and load trained weights
    lie_poses_mb = []
    for k in range(args.n_object):
        lie_poses = LiePose(n_view=n_view)
        weight_path_pose = osp.join(exp_base, 'pose_%06d_%02d.pth.tar' % (args.checkpoint, k))
        lie_poses.load_state_dict(torch.load(weight_path_pose))
        lie_poses_mb.append(lie_poses)

    # Load probability field for object scales
    weight_path_scale = osp.join(args.exp_base, 'sample%d/%d.pth.tar'%(args.n_sample_scale_test, args.scale_id))
    scale_var = torch.load(weight_path_scale)


    """
    Load Colmap results
    """
    colmap_visibles = []
    for k in range(args.n_object):
        colmap_vis = np.load(osp.join(args.data_root, args.preproc_path, 'poses/vis%d.npy' % (k)))
        colmap_visibles.append(colmap_vis)
    colmap_visibles = np.stack(colmap_visibles, 0)  # (K, Nv)
    colmap_visibles = torch.Tensor(colmap_visibles)

    # Load initialized object scale ranges
    scale_ranges = []
    for k in range(1, args.n_object):
        scale_range = np.load(osp.join(args.data_root, args.preproc_path, 'depth/scale%d.npy' % (k)))
        scale_ranges.append(scale_range)
    scale_ranges = np.stack(scale_ranges, 0)
    scale_ranges = torch.Tensor(scale_ranges)

    # Estimate current object scales
    scales = (scale_ranges[:, 1] - scale_ranges[:, 0]) * scale_var + scale_ranges[:, 0]
    bg_scale = torch.Tensor([1.])  # Append the fixed BG scale for convenience
    scales = torch.cat([bg_scale, scales], 0)


    save_path = osp.join(exp_base, 'sample%d/%d'%(args.n_sample_scale_test, args.scale_id))

    if args.render_test:
        """
        Render test views
        """
        save_rgb_path = osp.join(save_path, 'images_test')
        os.makedirs(save_rgb_path, exist_ok=True)
        save_depth_path = osp.join(save_path, 'depth_test')
        os.makedirs(save_depth_path, exist_ok=True)
        save_segm_path = osp.join(save_path, 'segm_test')
        os.makedirs(save_segm_path, exist_ok=True)
        save_depth_vis_path = osp.join(save_path, 'depth_vis_test')
        os.makedirs(save_depth_vis_path, exist_ok=True)
        save_segm_vis_path = osp.join(save_path, 'segm_vis_test')
        os.makedirs(save_segm_vis_path, exist_ok=True)

        # Decompose object pose from cam2obj pose (Known camera pose bundled to the first object)
        poses_mb = [lie_poses.get_all_poses().detach() for lie_poses in lie_poses_mb]
        poses_mb = torch.stack(poses_mb, 0)  # (K, Nv, 4, 4)
        # Apply scales to camera-to-object poses (translations)
        poses_mb[:, :, :3, 3] *= scales.unsqueeze(-1).unsqueeze(-1)
        cam_poses = poses_mb[0]  # (Nv, 4, 4)
        obj_poses = decompose_cam_pose(cam_poses, poses_mb.transpose(0, 1)).transpose(0, 1)  # (K, Nv, 4, 4)

        # Load testing camera poses
        if args.dataset_type == 'indoor':
            # Testing views share cameras with training views
            cam_poses_test = np.load(osp.join(args.data_root, args.preproc_path, 'poses/pose0.npy'))
            cam_poses_test = torch.Tensor(cam_poses_test)   # (Nv, 4, 4)
        else:
            cam_poses_test = test_set.load_all_poses()
            cam_poses_test = torch.Tensor(cam_poses_test)   # (Nv, 4, 4)

        tbar = tqdm(total=n_test_sample)
        pose_zero = torch.eye(4)
        for sample_id in range(n_test_sample):
            # Get corresponding view and frame ids
            sel_view, sel_time = test_set.frames[sample_id]['view'], test_set.frames[sample_id]['time_step']
            visibles = colmap_visibles[:, sel_time]     # (K)

            # Compose cam2obj pose at specific camera view and time step
            cam_pose_test = cam_poses_test[sel_view:(sel_view+1)].expand(n_view, 4, 4)  # (1, 4, 4)
            obj_pose = obj_poses[:, sel_time:(sel_time+1)]  # (K, 1, 4, 4)
            poses_mb = compose_cam_pose(cam_pose_test, obj_pose.transpose(0, 1)).transpose(0, 1)  # (K, 1, 4, 4)
            poses_mb = poses_mb[:, 0]  # (K, 4, 4)

            # Get rays for all pixels
            cam = Camera(img_h, img_w, focal, pose=pose_zero)
            rays_o, rays_d = cam.get_rays()
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)

            # Batchify
            rgb_map, depth_map, segm_map = [], [], []
            for i in range(0, rays_o.shape[0], args.chunk_ray):
                # Forward
                with torch.no_grad():
                    rays_o_batch = rays_o[i:(i + args.chunk_ray)]
                    rays_d_batch = rays_d[i:(i + args.chunk_ray)]
                    viewdirs_batch = viewdirs[i:(i + args.chunk_ray)]
                    rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                                n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                    poses_batch = poses_mb.unsqueeze(1).expand(args.n_object, rays_o_batch.shape[0], 4, 4)  # (K, Nr, 4, 4)
                    visibles_batch = visibles.unsqueeze(1).expand(args.n_object, rays_o_batch.shape[0])       # (K, Nr)
                    ret_dict = nerf_render_composetest(rays,
                                                       scales,
                                                       models,
                                                       bboxes=bboxes,
                                                       poses_mb=poses_batch,
                                                       visibles=visibles_batch,
                                                       white_bkgd=args.white_bkgd)

                rgb_map.append(ret_dict['rgb_map'])
                depth_map.append(ret_dict['depth_map'])
                segm_map.append(ret_dict['segm_map'])

            rgb_rend = torch.cat(rgb_map, 0).reshape(int(img_h), int(img_w), 3)
            depth_rend = torch.cat(depth_map, 0).reshape(int(img_h), int(img_w))
            segm_rend = torch.cat(segm_map, 0).reshape(int(img_h), int(img_w), args.n_object)

            rgb_map = rgb_rend.cpu().numpy().clip(0., 1.)
            rgb_map = (255 * rgb_map).astype(np.uint8)
            imageio.imwrite(osp.join(save_rgb_path, '%04d.png' % (sample_id)), rgb_map)
            depth_map = (depth_rend / far).cpu().numpy().clip(0., 1.)
            np.save(osp.join(save_depth_path, '%04d.npy' % (sample_id)), depth_map)
            depth_map = (255 * (1. - depth_map)).astype(np.uint8)
            imageio.imwrite(osp.join(save_depth_vis_path, '%04d.png' % (sample_id)), depth_map)
            segm_map = segm_rend.cpu().numpy()
            segm_map = segm_map.argmax(-1)
            np.save(osp.join(save_segm_path, '%04d.npy' % (sample_id)), segm_map)
            segm_map = build_segm_vis(segm_map)
            segm_map = (255 * segm_map).astype(np.uint8)
            imageio.imwrite(osp.join(save_segm_vis_path, '%04d.png' % (sample_id)), segm_map)

            tbar.update(1)

    else:
        """
        Render training views
        """
        save_rgb_path = osp.join(save_path, 'images_train')
        os.makedirs(save_rgb_path, exist_ok=True)
        save_depth_path = osp.join(save_path, 'depth_train')
        os.makedirs(save_depth_path, exist_ok=True)
        save_segm_path = osp.join(save_path, 'segm_train')
        os.makedirs(save_segm_path, exist_ok=True)
        save_depth_vis_path = osp.join(save_path, 'depth_vis_train')
        os.makedirs(save_depth_vis_path, exist_ok=True)
        save_segm_vis_path = osp.join(save_path, 'segm_vis_train')
        os.makedirs(save_segm_vis_path, exist_ok=True)

        tbar = tqdm(total=n_view)
        pose_zero = torch.eye(4)
        for sel_view in range(n_view):
            visibles = colmap_visibles[:, sel_view]     # (K)

            # Get rays for all pixels
            cam = Camera(img_h, img_w, focal, pose=pose_zero)
            rays_o, rays_d = cam.get_rays()
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)

            # Batchify
            rgb_map, depth_map, segm_map = [], [], []
            for i in range(0, rays_o.shape[0], args.chunk_ray):
                # Forward
                with torch.no_grad():
                    rays_o_batch = rays_o[i:(i + args.chunk_ray)]
                    rays_d_batch = rays_d[i:(i + args.chunk_ray)]
                    viewdirs_batch = viewdirs[i:(i + args.chunk_ray)]
                    rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                                n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                    poses_batch = [lie_poses.get_pose(sel_view) for lie_poses in lie_poses_mb]
                    poses_batch = torch.stack(poses_batch, 0)  # (K, 4, 4)
                    poses_batch = poses_batch.unsqueeze(1).expand(args.n_object, rays_o_batch.shape[0], 4,
                                                                  4)  # (K, Nr, 4, 4)
                    visibles_batch = visibles.unsqueeze(1).expand(args.n_object, rays_o_batch.shape[0])       # (K, Nr)
                    ret_dict = nerf_render_compose(rays,
                                                   scales,
                                                   models,
                                                   bboxes=bboxes,
                                                   poses_mb=poses_batch,
                                                   visibles=visibles_batch,
                                                   white_bkgd=args.white_bkgd)

                rgb_map.append(ret_dict['rgb_map'])
                depth_map.append(ret_dict['depth_map'])
                segm_map.append(ret_dict['segm_map'])

            rgb_rend = torch.cat(rgb_map, 0).reshape(int(img_h), int(img_w), 3)
            depth_rend = torch.cat(depth_map, 0).reshape(int(img_h), int(img_w))
            segm_rend = torch.cat(segm_map, 0).reshape(int(img_h), int(img_w), args.n_object)

            rgb_map = rgb_rend.cpu().numpy().clip(0., 1.)
            rgb_map = (255 * rgb_map).astype(np.uint8)
            imageio.imwrite(osp.join(save_rgb_path, '%04d.png' % (sel_view)), rgb_map)
            depth_map = (depth_rend / far).cpu().numpy().clip(0., 1.)
            np.save(osp.join(save_depth_path, '%04d.npy' % (sel_view)), depth_map)
            depth_map = (255 * (1. - depth_map)).astype(np.uint8)
            imageio.imwrite(osp.join(save_depth_vis_path, '%04d.png' % (sel_view)), depth_map)
            segm_map = segm_rend.cpu().numpy()
            segm_map = segm_map.argmax(-1)
            np.save(osp.join(save_segm_path, '%04d.npy' % (sel_view)), segm_map)
            segm_map = build_segm_vis(segm_map)
            segm_map = (255 * segm_map).astype(np.uint8)
            imageio.imwrite(osp.join(save_segm_vis_path, '%04d.png' % (sel_view)), segm_map)

            tbar.update(1)


print('Good')