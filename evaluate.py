import argparse
import os
import os.path as osp
import math
import lpips
import numpy as np
import imageio.v3 as imageio

import torch
import torch.nn.functional as F


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def im2tensor(img):
    return torch.Tensor(img.transpose(0, 3, 1, 2) / 127.5 - 1.0)


def compute_psnr(img_true, img, mask):
    """
    :param img_true: (B, H, W, 3)
    :param img: (B, H, W, 3)
    :param mask: (B, H, W)
    :return:
        psnr: (B,)
    """
    batch_size = img_true.shape[0]
    img_true = img_true.astype(np.float32) / 255.
    img = img.astype(np.float32) / 255.
    mse = (img_true - img) ** 2

    mask = np.expand_dims(mask, axis=-1)
    mask = np.broadcast_to(mask, mse.shape)
    mse, mask = mse.reshape(batch_size, -1), mask.reshape(batch_size, -1)
    mse = (mse * mask).sum(1) / mask.sum(1).clip(1e-10)
    psnr = - 10. * np.log(mse) / np.log(10.)
    return psnr


def gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()

def create_window(w_size, channel=1):
    _1D_window = gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window

def compute_ssim(img_true, img, mask,
                 max_val=1.0, filter_size=11, k1=0.01, k2=0.03):
    """
    :param img_true: (B, H, W, 3)
    :param img: (B, H, W, 3)
    :param mask: (B, H, W)
    :return:
        ssim: (B,)
    """
    batch_size = img_true.shape[0]
    img_true = img_true.astype(np.float32) / 255.
    img = img.astype(np.float32) / 255.
    mask = np.expand_dims(mask, axis=-1)
    mask = np.broadcast_to(mask, img.shape)

    img_true, img, mask = img_true.transpose(0, 3, 1, 2), img.transpose(0, 3, 1, 2), mask.transpose(0, 3, 1, 2)   # (B, 3, H, W)
    img_true, img, mask = torch.Tensor(img_true), torch.Tensor(img), torch.Tensor(mask)

    padd = 0
    (_, channel, height, width) = img.size()
    window = create_window(filter_size, channel=channel)

    mu1 = F.conv2d(img * mask, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img_true * mask, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img**2 * mask, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img_true**2 * mask, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img * img_true * mask, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ssim_map = ssim_map.reshape(batch_size, -1)
    ssim_map = ssim_map.cpu().numpy()
    ssim = ssim_map.mean(1)
    return ssim


def compute_lpips(img_true, img, mask, lpips_loss):
    """
    :param img_true: (B, H, W, 3)
    :param img: (B, H, W, 3)
    :param mask: (B, H, W)
    :return:
        lpips_score: (B,)
    """
    batch_size = img_true.shape[0]
    mask = np.expand_dims(mask, axis=-1)
    mask = np.broadcast_to(mask, img_true.shape)

    img_true = im2tensor(img_true * mask)   # (B, 3, H, W)
    img = im2tensor(img * mask)  # (B, 3, H, W)
    lpips_score = lpips_loss.forward(img_true, img) # (B, 1, H, W)
    lpips_score = lpips_score.detach().cpu().numpy()[:, 0]  # (B, H, W)

    mask = mask[..., 0]
    lpips_score, mask = lpips_score.reshape(batch_size, -1), mask.reshape(batch_size, -1)
    lpips_score = (lpips_score * mask).sum(1) / mask.sum(1).clip(1e-10)
    return lpips_score


def calculate_metrics(dataset_files, render_files, covis_files, lpips_loss, white_bkgd=False):
    imgs, imgs_true, covis_masks = [], [], []
    for idx in range(len(dataset_files)):
        dataset_file, render_file = dataset_files[idx], render_files[idx]
        img_true = imageio.imread(dataset_file)
        if img_true.shape[2] > 3:
            if white_bkgd:
                img_true = img_true[..., :3] * img_true[..., 3:] + (1. - img_true[..., 3:])
            else:
                img_true = img_true[..., :3]
        imgs_true.append(img_true)

        img = imageio.imread(render_file)
        imgs.append(img)

        if len(covis_files) > 0:
            covis_file = covis_files[idx]
            covis_mask = imageio.imread(covis_file)
            covis_mask = (covis_mask > 0).astype(np.float32)
        else:
            covis_mask = np.ones(img_true.shape[:2])
        covis_masks.append(covis_mask)

    imgs_true, imgs, covis_masks = np.stack(imgs_true, 0), np.stack(imgs, 0), np.stack(covis_masks, 0)
    PSNRs = compute_psnr(imgs_true, imgs, covis_masks)
    SSIMs = compute_ssim(imgs_true, imgs, covis_masks)
    LPIPSs = compute_lpips(imgs_true, imgs, covis_masks, lpips_loss)

    PSNR, SSIM, LPIPS = np.mean(PSNRs), np.mean(SSIMs), np.mean(LPIPSs)

    return PSNR, SSIM, LPIPS


def compute_depth_loss(dyn_depth, gt_depth, mask):
    dyn_depth, gt_depth = dyn_depth.reshape(-1), gt_depth.reshape(-1)
    mask = mask.reshape(-1)
    dyn_depth, gt_depth = dyn_depth[mask > 0], gt_depth[mask > 0]

    # Compute depth loss at GT depth space
    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d.clamp(1e-10)

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    dyn_depth = dyn_depth_norm * s_gt.clamp(1e-10) + t_gt
    return torch.abs(dyn_depth - gt_depth).mean(), ((dyn_depth - gt_depth) ** 2).mean()


def calculate_depth_metrics(dataset_files, render_files, covis_files):
    MAEs, MSEs = [], []
    for idx in range(len(dataset_files)):
        dataset_file, render_file = dataset_files[idx], render_files[idx]
        gt_depth = np.load(dataset_file)
        pred_depth = np.load(render_file)

        if len(covis_files) > 0:
            covis_file = covis_files[idx]
            covis_mask = imageio.imread(covis_file)
            covis_mask = (covis_mask > 0).astype(np.float32)
        else:
            covis_mask = np.ones(gt_depth.shape)

        MAE, MSE = compute_depth_loss(torch.Tensor(pred_depth), torch.Tensor(gt_depth), torch.Tensor(covis_mask))
        MAEs.append(MAE.item())
        MSEs.append(MSE.item())

    MAE = np.mean(MAEs)
    MSE = np.mean(MSEs)
    return MAE, MSE


def panoptic_quality(segms_pred, segms_gt, masks):
    IoUs, TP, FP, FN = [], [], [], []
    for idx in range(len(segms_pred)):
        segm_pred, segm_gt, mask = segms_pred[idx], segms_gt[idx], masks[idx]
        segm_pred, segm_gt = segm_pred.reshape(-1), segm_gt.reshape(-1)
        mask = mask.reshape(-1)
        segm_pred, segm_gt = segm_pred[mask > 0], segm_gt[mask > 0]

        n_object = int(max(torch.unique(segm_gt).max(), torch.unique(segm_pred).max())) + 1
        # Convert to one hot
        segm_gt = torch.eye(n_object)[segm_gt.long()]   # (N, K)
        segm_pred = torch.eye(n_object)[segm_pred.long()]   # (N, K)

        intersection = torch.matmul(segm_gt.transpose(0, 1), segm_pred)  # (K, K)
        union = (segm_gt.unsqueeze(2) + segm_pred.unsqueeze(1)).sum(0) - intersection   # (K, K)
        iou = intersection / union.clamp(min=1e-6)  # (K, K)

        # In panoptic segmentation, Greedy gives the same result as Hungarian matching
        valid_n_pred, valid_n_gt = (segm_pred.sum(0) > 0).float().sum(), (segm_gt.sum(0) > 0).float().sum()
        iou = iou.max(dim=0)[0]

        tp = (iou >= 0.5).float().sum()
        iou = iou[iou >= 0.5].sum()
        fp = valid_n_pred - tp
        fn = valid_n_gt - tp
        IoUs.append(iou)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)

    PQ = torch.Tensor(IoUs).sum() / (torch.Tensor(TP).sum() + 0.5 * torch.Tensor(FP).sum() + 0.5 * torch.Tensor(FN).sum())
    return PQ.item()


def mIoU(segm_pred, segm_gt, mask):
    segm_pred, segm_gt = segm_pred.reshape(-1), segm_gt.reshape(-1)
    mask = mask.reshape(-1)
    segm_pred, segm_gt = segm_pred[mask > 0], segm_gt[mask > 0]

    # n_object = max(torch.unique(segm_gt).shape[0], torch.unique(segm_pred).shape[0])
    n_object = int(max(torch.unique(segm_gt).max(), torch.unique(segm_pred).max())) + 1
    # Convert to one hot
    segm_gt = torch.eye(n_object)[segm_gt.long()]   # (N, K)
    segm_pred = torch.eye(n_object)[segm_pred.long()]   # (N, K)

    intersection = torch.sum(segm_gt * segm_pred, dim=0)    # (K)
    union = torch.sum(segm_gt + segm_pred, dim=0) - intersection    # (K)
    iou = intersection / union.clamp(min=1e-6)

    valid_n_gt = (segm_gt.sum(0) > 0).float().sum()
    iou = iou.sum() / valid_n_gt.clamp(min=1e-6)
    return iou


def calculate_segm_metrics(dataset_files, render_files, covis_files):
    # Panoptic quality
    segms_pred, segms_gt, covis_masks = [], [], []
    for idx in range(len(dataset_files)):
        dataset_file, render_file = dataset_files[idx], render_files[idx]
        gt_segm = np.load(dataset_file)
        segms_gt.append(torch.Tensor(gt_segm))

        pred_segm = np.load(render_file)
        segms_pred.append(torch.Tensor(pred_segm))

        if len(covis_files) > 0:
            covis_file = covis_files[idx]
            covis_mask = imageio.imread(covis_file)
            covis_mask = (covis_mask > 0).astype(np.float32)
        else:
            covis_mask = np.ones(gt_segm.shape)

        covis_masks.append(torch.Tensor(covis_mask))
    PQ = panoptic_quality(segms_pred, segms_gt, covis_masks)

    # mIoU
    IoUs = []
    for idx in range(len(dataset_files)):
        dataset_file, render_file = dataset_files[idx], render_files[idx]
        gt_segm = np.load(dataset_file)
        pred_segm = np.load(render_file)

        if len(covis_files) > 0:
            covis_file = covis_files[idx]
            covis_mask = imageio.imread(covis_file)
            covis_mask = (covis_mask > 0).astype(np.float32)
        else:
            covis_mask = np.ones(gt_segm.shape)

        IoU = mIoU(torch.Tensor(pred_segm), torch.Tensor(gt_segm), torch.Tensor(covis_mask))
        IoUs.append(IoU.item())
    IoU = np.mean(IoUs)

    return PQ, IoU


def collect_files(data_root, prefix, split='train', postfix='png'):
    if split is not None:
        data_path = osp.join(data_root, prefix + '_' + split)
    else:
        data_path = osp.join(data_root, prefix)
    filenames = sorted(os.listdir(data_path))
    return [osp.join(data_path, filename) for filename in filenames if filename.endswith(postfix)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--render_path", type=str, help="Rendering result path")
    parser.add_argument("--split", type=str, default='test', help="train / test")
    parser.add_argument("--mask", action='store_true', help="Mask out regions not co-visible in training")
    parser.add_argument("--eval_depth", action='store_true', help="Evaluate depth renderings")
    parser.add_argument("--eval_segm", action='store_true', help="Evaluate segmentation renderings")
    args = parser.parse_args()

    print("Start evaluation...")
    print("Dataset path: %s"%(args.dataset_path))
    print("Rendering result path: %s"%(args.render_path))
    print("Split: %s"%(args.split))


    # Co-visibility mask file paths
    if args.mask and args.split == 'test':
        covis_files = collect_files(args.dataset_path, 'covis_mask', None, 'png')
    else:
        covis_files = []


    # Evaluate RGB renderings
    dataset_files = collect_files(args.dataset_path, 'images', args.split, 'png')
    render_files = collect_files(args.render_path, 'images', args.split, 'png')
    white_bkgd = False

    lpips_loss = lpips.LPIPS(net='alex', spatial=True)
    PSNR, SSIM, LPIPS = calculate_metrics(dataset_files, render_files, covis_files, lpips_loss, white_bkgd=white_bkgd)
    metric = {'PSNR': PSNR, 'SSIM': SSIM, 'LPIPS': LPIPS}


    # Evaluate depth renderings
    if args.eval_depth:
        dataset_files = collect_files(args.dataset_path, 'depth', args.split, 'npy')
        render_files = collect_files(args.render_path, 'depth', args.split, 'npy')

        MAE, MSE = calculate_depth_metrics(dataset_files, render_files, covis_files)
        metric['MAE'] = MAE
        metric['MSE'] = MSE


    # Evaluate segmentation renderings
    if args.eval_segm:
        dataset_files = collect_files(args.dataset_path, 'segm', args.split, 'npy')
        render_files = collect_files(args.render_path, 'segm', args.split, 'npy')

        PQ, IoU = calculate_segm_metrics(dataset_files, render_files, covis_files)
        metric['PQ'] = PQ
        metric['IoU'] = IoU


    # Output results
    print(metric)
    if args.mask:
        with open(osp.join(args.render_path, 'metrics_%s_masked.txt'%(args.split)), 'w') as f:
            f.write(str(metric))
    else:
        with open(osp.join(args.render_path, 'metrics_%s.txt'%(args.split)), 'w') as f:
            f.write(str(metric))


print('Good')