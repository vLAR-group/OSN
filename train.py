import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from matplotlib import pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from osn import ObjectScaleNet
from tensorf.tensoRF import TensorVMSplit
from tensorf.utils import N_to_reso, TVLoss, cal_n_samples
from pose import LiePose, compose_cam_pose
from camera import Camera, BatchCameras, Rays
from renders.render import nerf_render
from renders.render_compose import nerf_render as nerf_render_compose
from renders.render_mb import nerf_render as nerf_render_mb
from renders.render_zbuffer import zbuffer_render
from metric import mse_to_psnr, pixel_acc, absolute_traj_error
from utils.visual_util import build_segm_vis


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def extract_color_by_coords(imgs, coords, vids):
    """
    :param imgs: (Nv, H, W, 3) torch.Tensor.
    :param coords: (Nr, 2) torch.Tensor.
    :param vids: (Nr,) torch.Tensor.
    :return:
        color: (Nr, 3) torch.Tensor.
    """
    u, v = coords[:, 0], coords[:, 1]
    u_floor, v_floor = u.floor().long(), v.floor().long()
    color1, w1 = imgs[vids, v_floor, u_floor], (u_floor + 1 - u) * (v_floor + 1 - v)
    color2, w2 = imgs[vids, v_floor, u_floor + 1], (u - u_floor) * (v_floor + 1 - v)
    color3, w3 = imgs[vids, v_floor + 1, u_floor], (u_floor + 1 - u) * (v - v_floor)
    color4, w4 = imgs[vids, v_floor + 1, u_floor + 1], (u - u_floor) * (v - v_floor)
    color = color1 * w1.unsqueeze(1) + color2 * w2.unsqueeze(1) + color3 * w3.unsqueeze(1) + color4 * w4.unsqueeze(1)
    return color


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--use_wandb', dest='use_wandb', default=False, action='store_true', help='Use WANDB for logging')
    parser.add_argument('--checkpoint', type=int, default=0, help='Checkpoint (iteration) to resume training')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Create wandb logger
    if args.use_wandb:
        wandb.init(project=args.wandb['project'],
                   name=args.wandb['name'],
                   config=configs,
                   notes=args.wandb['notes'])

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
        n_view_train = train_set.n_sample
        print('%d training views'%(n_view_train))

        img_h, img_w, focal = train_set.img_h, train_set.img_w, train_set.focal
        imgs_train, segms_train, poses_train = \
            train_set.load_all_images(), train_set.load_all_segms(), train_set.load_all_poses()
        imgs_train, segms_train, poses_train = \
            torch.Tensor(imgs_train), torch.Tensor(segms_train), torch.Tensor(poses_train)
        poses_train = compose_cam_pose(poses_train[:, 0], poses_train[:, 1:])  # (Nv, K, 4, 4)
        poses_train = poses_train.permute(1, 0, 2, 3)  # (K, Nv, 4, 4)

        depths_train = train_set.load_all_depths()
        depths_train = torch.Tensor(depths_train)

        # Define bounds
        near, far = args.near, args.far
        near_far = torch.Tensor([near, far])
        print('near: %.3f, far: %.3f'%(near, far))

    elif args.dataset_type in ['dynamic_scene', 'multimotion', 'iphone', 'kitti']:
        from datasets.realworld import RealWorldDataset
        train_set = RealWorldDataset(data_root=args.data_root,
                                     split='train',
                                     segm_sam=True)

        n_view_train = train_set.n_sample
        print('%d training views'%(n_view_train))

        img_h, img_w, focal = train_set.img_h, train_set.img_w, train_set.focal
        imgs_train, segms_train = train_set.load_all_images(), train_set.load_all_segms()
        imgs_train, segms_train = torch.Tensor(imgs_train), torch.Tensor(segms_train)

        # Define bounds
        near, far = args.near, args.far
        near_far = torch.Tensor([near, far])
        print('near: %.3f, far: %.3f' % (near, far))

    else:
        raise ValueError('Not implemented!')


    """
    Create networks
    """
    # Load Bounding box for each object
    bboxes = []
    for k in range(args.n_object):
        bbox = np.load(osp.join(args.data_root, args.preproc_path, 'poses/bbox%d.npy' % (k)))
        bbox = torch.Tensor(bbox)
        bboxes.append(bbox)

    # TesnsoRF coarse-to-fine upsampling: linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(
        np.log(args.tensorf['N_voxel_init']), np.log(args.tensorf['N_voxel_final']), len(args.tensorf['upsamp_list']) + 1
    ))).long()).tolist()[1:]

    # Create the scale-invariant object representation (TensoRF) models
    models, grad_vars = [], []
    n_sample_points = []
    for k in range(args.n_object):
        aabb = bboxes[k].transpose(0, 1)
        reso_cur = N_to_reso(args.tensorf['N_voxel_init'], aabb)
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
        models.append(model)
        grad_vars += list(model.parameters())
        n_sample_points.append(cal_n_samples(reso_cur, step_ratio=args.step_ratio))

    if args.n_sample_point_adjust:
        n_sample_point = max(n_sample_points)
    else:
        n_sample_point = args.n_sample_point
    print('Set n_sample_point (per ray) to %d'%(n_sample_point))

    # Create the pose parameters
    lie_poses_mb, pose_vars = [], []
    for k in range(args.n_object):
        lie_poses = LiePose(n_view=n_view_train)
        lie_poses_mb.append(lie_poses)
        pose_vars += list(lie_poses.parameters())

    # Create the object scale net
    osn = ObjectScaleNet(input_dim=args.n_object-1,
                         n_dim=args.osn['n_dim'],
                         n_layer=args.osn['n_layer'],
                         alpha=args.osn['alpha'],
                         shift=args.osn['shift'])

    # Create the loss & optimizer
    mse_loss = nn.MSELoss(reduction='mean')
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    tvreg = TVLoss()
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_pose = torch.optim.Adam(params=pose_vars, lr=args.lrate_pose, betas=(0.9, 0.999))
    optimizer_osn = torch.optim.Adam(params=osn.parameters(), lr=args.lrate_osn, betas=(0.9, 0.999))

    # Create checkpoint path
    exp_base = args.exp_base
    os.makedirs(exp_base, exist_ok=True)


    """
    Load Colmap results
    """
    # Load Colmap camera poses
    colmap_visibles = []
    for k in range(args.n_object):
        colmap_poses = np.load(osp.join(args.data_root, args.preproc_path, 'poses/pose%d.npy' % (k)))
        colmap_poses = torch.Tensor(colmap_poses)
        lie_poses_mb[k].load_base_poses(colmap_poses)

        colmap_vis = np.load(osp.join(args.data_root, args.preproc_path, 'poses/vis%d.npy' % (k)))
        colmap_visibles.append(colmap_vis)
    colmap_visibles = np.stack(colmap_visibles, 0)   # (K, Nv)
    colmap_visibles = torch.Tensor(colmap_visibles)

    # Load Colmap depths
    colmap_depths, colmap_depth_weights, colmap_depth_vids = [], [], []
    for k in range(args.n_object):
        colmap_depth = np.load(osp.join(args.data_root, args.preproc_path, 'depth/depth%d.npy' % (k)))
        colmap_depth, colmap_depth_weight = colmap_depth[:, :3], colmap_depth[:, 4]

        # Remove points out of image boundary
        valid_w1 = (colmap_depth[:, 0] >= 0)
        valid_w2 = (colmap_depth[:, 0] < (img_w - 1))
        valid_h1 = (colmap_depth[:, 1] >= 0)
        valid_h2 = (colmap_depth[:, 1] < (img_h - 1))
        valid = valid_w1 * valid_w2 * valid_h1 * valid_h2
        colmap_depth, colmap_depth_weight = colmap_depth[valid], colmap_depth_weight[valid]

        colmap_depth, colmap_depth_weight = torch.Tensor(colmap_depth), torch.Tensor(colmap_depth_weight)
        colmap_depths.append(colmap_depth)
        colmap_depth_weights.append(colmap_depth_weight)

        colmap_depth_vid = np.load(osp.join(args.data_root, args.preproc_path, 'depth/vid%d.npy' % (k)))
        colmap_depth_vid = colmap_depth_vid[valid]
        colmap_depth_vids.append(colmap_depth_vid)

    # Load initialized object scale ranges
    scale_ranges = []
    for k in range(1, args.n_object):
        scale_range = np.load(osp.join(args.data_root, args.preproc_path, 'depth/scale%d.npy' % (k)))
        scale_ranges.append(scale_range)
    scale_ranges = np.stack(scale_ranges, 0)
    scale_ranges = torch.Tensor(scale_ranges)


    """
    Resume from previous training
    """
    if args.checkpoint > 0:
        print('Resume from checkpoint %d'%(args.checkpoint))
        exp_base = args.exp_base
        checkpoint = args.checkpoint

        for k in range(args.n_object):
            weight_path = osp.join(exp_base, 'model_%06d_%02d.pth.tar' % (checkpoint, k))
            models[k].load_state_dict(torch.load(weight_path))
            weight_path_pose = osp.join(exp_base, 'pose_%06d_%02d.pth.tar' % (checkpoint, k))
            lie_poses_mb[k].load_state_dict(torch.load(weight_path_pose))

        weight_path = osp.join(exp_base, 'scale_%06d.pth.tar' % (checkpoint))
        osn.load_state_dict(torch.load(weight_path))

        # optim_path = osp.join(exp_base, 'optim_%06d.pth.tar' % (checkpoint))
        # optimizer.load_state_dict(torch.load(optim_path))
        # optim_path_pose = osp.join(exp_base, 'optim_pose_%06d.pth.tar' % (checkpoint))
        # optimizer_pose.load_state_dict(torch.load(optim_path_pose))
        # optim_path_scale = osp.join(exp_base, 'optim_scale_%06d.pth.tar' % (checkpoint))
        # optimizer_osn.load_state_dict(torch.load(optim_path_scale))

        # Decay learning rate
        new_lrate = args.lrate * (args.lrate_decay ** (checkpoint / args.lrate_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        new_lrate_pose = args.lrate_pose * (args.lrate_decay_pose ** (checkpoint / args.lrate_decay_step_pose))
        for param_group in optimizer_pose.param_groups:
            param_group['lr'] = new_lrate_pose


    """
    Training loop
    """
    tbar = tqdm(total=(args.n_iters - args.checkpoint))
    pose_zero = torch.eye(4)
    poses_zero = pose_zero.expand(n_view_train, 4, 4)

    for it in range(args.checkpoint+1, args.n_iters+1):
    # for it in range(args.checkpoint, args.n_iters+1):
        cams = BatchCameras(img_h, img_w, focal, poses=poses_zero)

        n_sample_ray = args.n_sample_ray
        n_sample_ray_color = int(0.5 * n_sample_ray)
        n_sample_ray_depth = n_sample_ray - n_sample_ray_color

        # Sample rays for color supervision
        segm_mask = torch.ones(n_view_train, int(img_h), int(img_w), dtype=torch.bool)
        rays_o, rays_d, select_coords = cams.sample_rays_with_mask(mask=segm_mask,
                                                                   n_sample_ray=(n_sample_ray_color * args.n_object))
        poses_mb = [lie_poses.get_all_poses()[select_coords[:, 0]] for lie_poses in lie_poses_mb]
        poses_mb = torch.stack(poses_mb, 0)
        target = imgs_train[select_coords[:, 0], select_coords[:, 1], select_coords[:, 2]]
        visibles = colmap_visibles[:, select_coords[:, 0]]

        # Sample rays for depth supervision
        rays_o_mb_depth, rays_d_mb_depth = [], []
        target_depth_mb, weight_depth_mb = [], []
        target_segm_mb = []
        poses_mb_depth = []
        target_color_mb = []
        visibles_depth_mb = []
        # Sample rays for each object separately
        for k in range(args.n_object):
            colmap_depth, colmap_depth_weight = colmap_depths[k], colmap_depth_weights[k]
            coords, target_depth = colmap_depth[:, :2], colmap_depth[:, 2]
            rays_o_depth, rays_d_depth, select_inds = cams.sample_rays_from_coords(coords=coords, n_sample_ray=n_sample_ray_depth)
            rays_o_mb_depth.append(rays_o_depth)
            rays_d_mb_depth.append(rays_d_depth)

            target_depth = target_depth[select_inds]
            target_depth_mb.append(target_depth)
            weight_depth = colmap_depth_weight[select_inds]
            weight_depth_mb.append(weight_depth)

            target_segm = k * torch.ones(rays_o_depth.shape[0], dtype=torch.long)
            target_segm_mb.append(target_segm)

            # Get the to-be-optimized poses
            colmap_depth_vid = colmap_depth_vids[k]
            colmap_depth_vid = colmap_depth_vid[select_inds]
            poses = [lie_poses.get_all_poses()[colmap_depth_vid] for lie_poses in lie_poses_mb]
            poses = torch.stack(poses, 0)
            poses_mb_depth.append(poses)

            # Extract RGB color at corresponding coordinates
            coords = coords[select_inds]
            target_color = extract_color_by_coords(imgs_train, coords, colmap_depth_vid)
            target_color_mb.append(target_color)

            # Extract object visibilities in the view of selected rays
            visibles_depth = colmap_visibles[:, colmap_depth_vid]
            visibles_depth_mb.append(visibles_depth)

        # Collect rays from all objects
        rays_o_depth, rays_d_depth = torch.cat(rays_o_mb_depth, 0), torch.cat(rays_d_mb_depth, 0)
        target_depth = torch.cat(target_depth_mb, 0)
        weight_depth = torch.cat(weight_depth_mb, 0)
        target_segm = torch.cat(target_segm_mb, 0)
        poses_mb_depth = torch.cat(poses_mb_depth, 1)
        target_color = torch.cat(target_color_mb, 0)
        visibles_depth = torch.cat(visibles_depth_mb, 1)

        # Rays for sparse depth & segm supervision
        viewdirs_depth = rays_d_depth / rays_d_depth.norm(dim=1, keepdim=True)
        rays_depth = Rays(rays_o_depth, rays_d_depth, viewdirs_depth, n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
        # Rays for dense color supervision
        viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
        rays = Rays(rays_o, rays_d, viewdirs, n_sample_point, args.n_sample_point_fine, near, far, args.perturb)

        if it <= args.n_iters_bootstrap:
            # Separate rendering of each object
            ret_dict_depth = nerf_render(rays_depth,
                                         models,
                                         bboxes=bboxes,
                                         poses_mb=poses_mb_depth,
                                         segm=target_segm,
                                         white_bkgd=args.white_bkgd)
        else:
            # First sample a valid scale by rejection sampling, then composite rendering of multiple objects
            scale_vars = torch.rand(args.osn['n_sample_scale'], args.n_object - 1)
            with torch.no_grad():
                scores = osn(scale_vars)
            scores = scores.squeeze(-1)
            scale_vars_valid = scale_vars[scores > args.osn['score_thresh']]
            scale_var = scale_vars_valid[0]

            scales = (scale_ranges[:, 1] - scale_ranges[:, 0]) * scale_var + scale_ranges[:, 0]
            bg_scale = torch.Tensor([1.])  # Append the fixed BG scale for convenience
            scales = torch.cat([bg_scale, scales], 0)

            ret_dict_depth = nerf_render_compose(rays_depth, 
                                                 scales,
                                                 models,
                                                 bboxes=bboxes,
                                                 poses_mb=poses_mb_depth,
                                                 visibles=visibles_depth,
                                                 white_bkgd=args.white_bkgd)

            ret_dict = nerf_render_compose(rays, 
                                           scales,
                                           models,
                                           bboxes=bboxes,
                                           poses_mb=poses_mb,
                                           visibles=visibles,
                                           white_bkgd=args.white_bkgd)

        # Image re-rendering loss
        if it <= args.n_iters_bootstrap:
            rgb_map = ret_dict_depth['rgb_map']
            loss_img = mse_loss(rgb_map, target_color)
        else:
            rgb_map = ret_dict['rgb_map']
            loss_img = mse_loss(rgb_map, target)
        psnr = mse_to_psnr(loss_img.detach().cpu())

        # Depth re-rendering loss
        depth_map = ret_dict_depth['depth_map']
        # Scale the depth
        if it > args.n_iters_bootstrap:
            target_depth *= scales[target_segm].detach()
        # Apply robust weighted depth supervision
        if args.loss_depth_weighted:
            loss_depth = ((depth_map - target_depth) ** 2)
            loss_depth = (weight_depth * loss_depth).sum() / weight_depth.sum()
        else:
            loss_depth = mse_loss(depth_map, target_depth)

        # Segmentation loss
        if it <= args.n_iters_bootstrap:
            loss_segm = torch.Tensor([0.])
        else:
            segm_map = ret_dict_depth['segm_map']
            # Normalize rendered segmentation map
            segm_map = segm_map / segm_map.sum(-1, keepdims=True).clamp(min=1e-8)
            loss_segm = ce_loss(segm_map, target_segm.long()).mean()

        # Entropy loss
        if it <= args.n_iters_bootstrap:
            loss_entropy = ret_dict_depth['entropy']
        else:
            loss_entropy = ret_dict['entropy']

        # Total variation loss
        loss_tv_density, loss_tv_app = 0, 0
        for model in models:
            loss_tv_density += model.TV_loss_density(tvreg)
            loss_tv_app += model.TV_loss_app(tvreg)

        # Total loss
        loss = args.loss_img_w * loss_img + args.loss_depth_w * loss_depth +\
               args.loss_segm_w * loss_segm + args.loss_entropy_w * loss_entropy +\
               args.tensorf['TV_weight_density'] * loss_tv_density + args.tensorf['TV_weight_app'] * loss_tv_app

        # Add train logs
        train_logs = {"train_loss": loss_img.item(), "train_psnr": psnr.item(),
                      "train_loss_depth": loss_depth.item(),
                      "train_loss_segm": loss_segm.item(),
                      "train_loss_entropy": loss_entropy.item(),
                      "train_loss_tv_density": loss_tv_density.item(),
                      "train_loss_tv_app": loss_tv_app.item()}

        # Monitor the camera pose estimation error
        if args.dataset_type == 'indoor':
            for k in range(args.n_object):
                with torch.no_grad():
                    poses = lie_poses_mb[k].get_all_poses()
                    R_error, t_error, _, _, _ = absolute_traj_error(poses.detach(), poses_train[k])

                # Add train logs
                train_log = {"train_R_error_%02d" % (k): R_error.item(), "train_t_error_%02d" % (k): t_error.item()}
                train_logs = train_logs | train_log

        # Backward
        optimizer.zero_grad()
        optimizer_pose.zero_grad()
        loss.backward()
        optimizer.step()
        if it > args.n_iters_bootstrap:
            optimizer_pose.step()

        # Save model checkpoint
        if it % args.save_freq == 0:
            for k in range(args.n_object):
                torch.save(models[k].state_dict(), osp.join(args.exp_base, 'model_%06d_%02d.pth.tar'%(it, k)))
                torch.save(lie_poses_mb[k].state_dict(), osp.join(args.exp_base, 'pose_%06d_%02d.pth.tar'%(it, k)))
            torch.save(osn.state_dict(), osp.join(args.exp_base, 'osn_%06d.pth.tar'%(it)))
            # # Also save optimizer states to resume training
            # torch.save(optimizer.state_dict(), osp.join(args.exp_base, 'optim_%06d.pth.tar'%(it)))
            # torch.save(optimizer_pose.state_dict(), osp.join(args.exp_base, 'optim_pose_%06d.pth.tar'%(it)))
            # torch.save(optimizer_osn.state_dict(), osp.join(args.exp_base, 'optim_osn_%06d.pth.tar'%(it)))

        # Upsample TensoRF resolution
        if it in args.tensorf['upsamp_list']:
            n_voxels = N_voxel_list.pop(0)
            grad_vars = []
            n_sample_points = []
            for model in models:
                reso_cur = N_to_reso(n_voxels, model.aabb)
                model.upsample_volume_grid(reso_cur)
                grad_vars += list(model.parameters())
                n_sample_points.append(cal_n_samples(reso_cur, step_ratio=args.step_ratio))
            optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
            if args.n_sample_point_adjust:
                n_sample_point = max(n_sample_points)
            print('Set n_sample_point (per ray) to %d'%(n_sample_point))

        # Decay learning rate
        new_lrate = args.lrate * (args.lrate_decay ** (it / args.lrate_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        train_logs['lrate'] = new_lrate

        new_lrate_pose = args.lrate_pose * (args.lrate_decay_pose ** (it / args.lrate_decay_step_pose))
        for param_group in optimizer_pose.param_groups:
            param_group['lr'] = new_lrate_pose
        train_logs['lrate_pose'] = new_lrate_pose


        # Train the object scale net
        if it >= args.n_iters_bootstrap and it % args.n_iters_tensorf == 0 and (it - args.n_iters_bootstrap) // args.n_iters_tensorf < args.n_alter_round:

            # Determine the thresh
            iters_thresh_list = args.osn['iters_thresh_list']
            iters_thresh_list = np.array(iters_thresh_list)
            valid = (it >= iters_thresh_list)
            iters_thresh_list = iters_thresh_list[valid]
            select_idx = len(iters_thresh_list) - 1
            acc_thresh = args.osn['acc_thresh_list'][select_idx]
            topk_thresh = args.osn['topk_thresh_list'][select_idx]

            sbar = tqdm(total=args.n_iters_osn)
            losses_score, losses_prior = [], []

            for it_osn in range(args.n_iters_osn):
                # Sample rays for depth supervision
                rays_o_mb_depth, rays_d_mb_depth = [], []
                target_segm_mb = []
                poses_mb_depth = []
                visibles_depth_mb = []
                # Sample rays for each object separately
                for k in range(args.n_object):
                    colmap_depth, colmap_depth_weight = colmap_depths[k], colmap_depth_weights[k]
                    coords, target_depth = colmap_depth[:, :2], colmap_depth[:, 2]
                    rays_o_depth, rays_d_depth, select_inds = cams.sample_rays_from_coords(coords=coords,
                                                                                           n_sample_ray=n_sample_ray_depth)
                    rays_o_mb_depth.append(rays_o_depth)
                    rays_d_mb_depth.append(rays_d_depth)

                    target_segm = k * torch.ones(rays_o_depth.shape[0], dtype=torch.long)
                    target_segm_mb.append(target_segm)

                    # Get the to-be-optimized poses
                    colmap_depth_vid = colmap_depth_vids[k]
                    colmap_depth_vid = colmap_depth_vid[select_inds]
                    poses = [lie_poses.get_all_poses()[colmap_depth_vid] for lie_poses in lie_poses_mb]
                    poses = torch.stack(poses, 0)
                    poses_mb_depth.append(poses)

                    # Extract object visibilities in the view of selected rays
                    visibles_depth = colmap_visibles[:, colmap_depth_vid]
                    visibles_depth_mb.append(visibles_depth)

                # Collect rays from all objects
                rays_o_depth, rays_d_depth = torch.cat(rays_o_mb_depth, 0), torch.cat(rays_d_mb_depth, 0)
                target_segm = torch.cat(target_segm_mb, 0)
                poses_mb_depth = torch.cat(poses_mb_depth, 1)
                visibles_depth = torch.cat(visibles_depth_mb, 1)

                # Rays for sparse depth & segm supervision
                viewdirs_depth = rays_d_depth / rays_d_depth.norm(dim=1, keepdim=True)
                rays_depth = Rays(rays_o_depth, rays_d_depth, viewdirs_depth, n_sample_point, args.n_sample_point_fine,
                                  near, far, args.perturb)

                # Sample many scales to optimize the probability field
                scale_vars = torch.rand(args.osn['n_sample_scale'], args.n_object - 1)
                scores = osn(scale_vars)
                scores = scores.squeeze(-1)
                scales = (scale_ranges[:, 1] - scale_ranges[:, 0]) * scale_vars + scale_ranges[:, 0]
                bg_scale = torch.Tensor([1.]).repeat(args.osn['n_sample_scale'], 1)  # Append the fixed BG scale for convenience
                scales = torch.cat([bg_scale, scales], 1)

                with torch.no_grad():
                    ret_dict_depth = nerf_render_mb(rays_depth,
                                                    models,
                                                    bboxes=bboxes,
                                                    poses_mb=poses_mb_depth,
                                                    visibles=visibles_depth,
                                                    white_bkgd=args.white_bkgd)

                # Soft Z-buffer rendering
                segm_map = zbuffer_render(ret_dict_depth['rgb_map_mb'],
                                          ret_dict_depth['depth_map_mb'],
                                          ret_dict_depth['acc_map_mb'],
                                          scales,
                                          zbuffer_beta=args.zbuffer_beta,
                                          render_segm_only=True)    # (Ns, Nr, K)

                # Compute segmentation metric
                target_segm = target_segm.repeat(args.osn['n_sample_scale'], 1)  # (Ns, Nr)
                acc = pixel_acc(segm_map, target_segm.long())

                valid = (acc > acc_thresh)
                n_sample_scale_thresh = int(topk_thresh * args.osn['n_sample_scale'])
                scores_gt = torch.zeros_like(scores)
                # Select the larger one in "mIoU > 0.95" and "mIoU in top 5%"
                if valid.sum() < n_sample_scale_thresh:
                    valid = torch.argsort(acc, descending=True)[:n_sample_scale_thresh]
                scores_gt[valid] = 1.

                loss_score = mse_loss(scores, scores_gt)
                loss_prior = - scores.mean()    # Only for monitoring

                losses_score.append(loss_score.item())
                losses_prior.append(loss_prior.item())

                # Backward
                optimizer_osn.zero_grad()
                loss_score.backward()
                optimizer_osn.step()

                sbar.update(1)

            train_logs['train_loss_score'] = loss_score.item()
            train_logs['train_loss_prior'] = loss_prior.item()

            # Save prob field training log
            fig = plt.figure(figsize=(10, 5))
            plt.plot(losses_score, label='loss_score')
            plt.plot(losses_prior, label='loss_prior')
            plt.legend()
            plt.savefig(osp.join(args.exp_base, 'osn_%06d.png'%(it)))


        # Validation
        if it % args.val_freq == 0:
            sel_view = (it // args.val_freq) % n_view_train
            target = imgs_train[sel_view]
            target_segm = segms_train[sel_view]
            if args.dataset_type == 'indoor':
                target_depth = depths_train[sel_view]
            visibles = colmap_visibles[:, sel_view]     # (K)

            # Get rays for all pixels
            cam = Camera(img_h, img_w, focal, pose=pose_zero)
            rays_o, rays_d = cam.get_rays()
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)

            with torch.no_grad():
                # Fix the probability field, sample a valid scale by rejection sampling
                scale_vars = torch.rand(args.osn['n_sample_scale'], args.n_object - 1)
                scores = osn(scale_vars)
                scores = scores.squeeze(-1)
                scale_vars_valid = scale_vars[scores > args.osn['score_thresh']]

                # Sample a random one if cannot get a valid scale
                if scale_vars_valid.shape[0] > 0:
                    scale_var = scale_vars_valid[0]
                else:
                    scale_var = scale_vars[0]

                scales = (scale_ranges[:, 1] - scale_ranges[:, 0]) * scale_var + scale_ranges[:, 0]
                bg_scale = torch.Tensor([1.])  # Append the fixed BG scale for convenience
                scales = torch.cat([bg_scale, scales], 0)

            # Batchify
            rgb_map, depth_map, segm_map = [], [], []
            for i in range(0, rays_o.shape[0], args.chunk_ray):
                # Forward
                with torch.no_grad():
                    rays_o_batch = rays_o[i:(i+args.chunk_ray)]
                    rays_d_batch = rays_d[i:(i+args.chunk_ray)]
                    viewdirs_batch = viewdirs[i:(i + args.chunk_ray)]
                    rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                                n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                    poses_batch = [lie_poses.get_pose(sel_view) for lie_poses in lie_poses_mb]
                    poses_batch = torch.stack(poses_batch, 0)       # (K, 4, 4)
                    poses_batch = poses_batch.unsqueeze(1).expand(args.n_object, rays_o_batch.shape[0], 4, 4)       # (K, Nr, 4, 4)
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

            rgb_rend = torch.cat(rgb_map, 0).reshape(target.shape)
            depth_rend = torch.cat(depth_map, 0).reshape(target.shape[0], target.shape[1])
            segm_rend = torch.cat(segm_map, 0).reshape(target.shape[0], target.shape[1], args.n_object)
            # Normalize rendered segmentation map
            segm_rend = segm_rend / segm_rend.sum(-1, keepdims=True).clamp(min=1e-8)

            loss_img = mse_loss(rgb_rend, target)
            psnr = mse_to_psnr(loss_img.detach().cpu())
            loss_segm = ce_loss(segm_rend.reshape(-1, args.n_object), target_segm.reshape(-1).long()).mean()
            val_log = {"val_loss": loss_img.item(), "val_psnr": psnr.item(),
                       "val_loss_segm": loss_segm.item()}
            if args.dataset_type == 'indoor':
                loss_depth = mse_loss(depth_rend, target_depth)
                val_log["val_loss_depth"] = loss_depth.item()
            train_logs = train_logs | val_log

            # Add validation logs
            rgb_map = rgb_rend.cpu().numpy().clip(0., 1.)
            rgb_map = wandb.Image(rgb_map, caption="coarse rendering")
            depth_map = (depth_rend / far).cpu().numpy().clip(0., 1.)
            depth_map = (255 * (1. - depth_map)).astype(np.uint8)
            depth_map = wandb.Image(depth_map, caption="coarse depth")
            segm_map = segm_rend.cpu().numpy().argmax(-1)
            segm_map = build_segm_vis(segm_map)
            segm_map = wandb.Image(segm_map, caption="coarse segm")
            render_log = {'val_img': rgb_map, 'val_depth': depth_map, 'val_segm': segm_map}
            if args.dataset_type == 'indoor':
                target_depth = (target_depth / far).cpu().numpy().clip(0., 1.)
                target_depth = (255 * (1. - target_depth)).astype(np.uint8)
                target_depth = wandb.Image(target_depth, caption="GT depth")
                render_log['gt_depth'] = target_depth
            train_logs = train_logs | render_log

        # Logging
        if args.use_wandb:
            wandb.log(train_logs)

        tbar.update(1)