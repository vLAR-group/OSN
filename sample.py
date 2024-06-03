import os
import os.path as osp
import yaml
import argparse
import numpy as np

import torch

from osn import ObjectScaleNet


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--checkpoint', type=int, default=200000, help='Checkpoint (iteration) to load')

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


    exp_base = args.exp_base

    # Load object scale net
    osn = ObjectScaleNet(input_dim=args.n_object - 1,
                         n_dim=args.osn['n_dim'],
                         n_layer=args.osn['n_layer'],
                         alpha=args.osn['alpha'],
                         shift=args.osn['shift'])
    weight_path = osp.join(exp_base, 'osn_%06d.pth.tar' % (args.checkpoint))
    osn.load_state_dict(torch.load(weight_path))
    osn.eval()

    # Load initialized object scale ranges
    scale_ranges = []
    for k in range(1, args.n_object):
        scale_range = np.load(osp.join(args.data_root, args.preproc_path, 'depth/scale%d.npy' % (k)))
        scale_ranges.append(scale_range)
    scale_ranges = np.stack(scale_ranges, 0)
    scale_ranges = torch.Tensor(scale_ranges)

    # Load GT scales
    if args.dataset_type == 'indoor':
        scales_gt = np.load(osp.join(args.data_root, args.preproc_path, 'depth/scales_gt.npy'))
        scales_gt = torch.Tensor(scales_gt)
        # Absolute -> relative GT scale
        scale_var_gt = scales_gt[1:] / scales_gt[0]
        scale_var_gt = (scale_var_gt - scale_ranges[:, 0]) / (scale_ranges[:, 1] - scale_ranges[:, 0])
    # # Load test-time optimized scales
    # else:
    #     weight_path_scale = osp.join(args.exp_base, 'scale_ttp_%06d.pth.tar' % (args.checkpoint))
    #     scale_var_gt = torch.load(weight_path_scale)


    n_sample_scale_test = 100   # 1000
    score_thresh = 0.5
    topk = 5
    n_sample_scale_train = 50

    # Sample object scales, keep scales with high probability
    scale_vars = torch.rand(500 * n_sample_scale_test, args.n_object - 1)
    with torch.no_grad():
        scores = osn(scale_vars)
    scores = scores.squeeze(-1)
    scale_vars = scale_vars[scores > score_thresh]
    scores = scores[scores > score_thresh]


    # For synthetic dataset, only render top-K closest scales to GT scale
    if args.dataset_type == 'indoor':
        select_ids = np.random.choice(scale_vars.shape[0], n_sample_scale_test, replace=False)
        scale_vars = scale_vars[select_ids]
        scores = scores[select_ids]

        # Take top-K closest scles to GT scale, save time for rendering
        errors = (scale_vars - scale_var_gt).norm(dim=-1)
        sort_idx = errors.argsort()
        topk_idx = sort_idx[:topk]
        scale_vars_topk = scale_vars[topk_idx]
        errors_topk = errors[topk_idx]
        print('Top-K closest distance of %d sampled scales to GT scale:' % (n_sample_scale_test), errors_topk)

        # Take another K scales close to GT scale, for robustness (closest scales to GT may not produce best results)
        robust_idx = sort_idx[topk:topk*3]
        select_ids = np.random.choice(robust_idx.shape[0], topk, replace=False)
        robust_idx = robust_idx[select_ids]
        scale_vars_robust = scale_vars[robust_idx]
        errors_robust = errors[robust_idx]
        print('Top-K robust distance of %d sampled scales to GT scale:' % (n_sample_scale_test), errors_robust)

        # Save selected scales
        scale_vars_test = torch.cat([scale_vars_topk, scale_vars_robust], 0)
        save_path = osp.join(args.exp_base, 'sample%d'%(n_sample_scale_test))
        os.makedirs(save_path, exist_ok=True)
        for k in range(scale_vars_test.shape[0]):
            scale_var = scale_vars_test[k]
            torch.save(scale_var, osp.join(save_path, '%d.pth.tar' % (k)))


    # Save all sampled scales for other datasets
    else:
        select_ids = np.random.choice(scale_vars.shape[0], n_sample_scale_test, replace=False)
        scale_vars = scale_vars[select_ids]

        # Save selected scales
        save_path = osp.join(args.exp_base, 'sample%d'%(n_sample_scale_test))
        os.makedirs(save_path, exist_ok=True)
        for k in range(scale_vars.shape[0]):
            scale_var = scale_vars[k]
            torch.save(scale_var, osp.join(save_path, '%d.pth.tar' % (k)))