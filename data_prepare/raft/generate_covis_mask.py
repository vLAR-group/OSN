import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as imageio

from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder

from flow_utils import *
from generate_flow import create_dir, load_image, resize_flow, warp_flow

DEVICE = "cuda"


def compute_fwd(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask


def run(args, input_path_src, input_path_target, output_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    images_src = glob.glob(os.path.join(input_path_src, "*.png"))
    images_src = sorted(images_src)
    images_target = glob.glob(os.path.join(input_path_target, "*.png"))
    images_target = sorted(images_target)

    create_dir(output_path)
    output_path_tmp = output_path + '_tmp'
    create_dir(output_path_tmp)

    # Threshold to for covisibility
    threshold = max(2.0, 0.1 * len(images_target))
    print('Threshold: ', threshold)
    # threshold = max(5.0, 0.1 * len(images_src))


    with torch.no_grad():
        img_train = cv2.imread(images_src[0])
        tbar = tqdm(range(len(images_src)))
        for i in range(len(images_src)):
            covis_masks = []
            for j in range(len(images_target)):
                if os.path.exists(os.path.join(output_path_tmp, '%04d_%04d.png'%(i, j))):
                    mask_fwd = imageio.imread(os.path.join(output_path_tmp, '%04d_%04d.png'%(i, j)))
                    mask_fwd = mask_fwd.astype(np.float32) / 255.0
                    covis_masks.append(mask_fwd)

                else:
                    image_src = load_image(images_src[i])
                    image_target = load_image(images_target[j])

                    padder = InputPadder(image_src.shape)
                    image_src, image_target = padder.pad(image_src, image_target)

                    _, flow_fwd = model(image_src, image_target, iters=20, test_mode=True)
                    _, flow_bwd = model(image_target, image_src, iters=20, test_mode=True)

                    flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
                    flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

                    flow_fwd = resize_flow(flow_fwd, img_train.shape[0], img_train.shape[1])
                    flow_bwd = resize_flow(flow_bwd, img_train.shape[0], img_train.shape[1])

                    mask_fwd = compute_fwd(flow_fwd, flow_bwd)
                    mask_fwd = mask_fwd.astype(np.float32)
                    covis_masks.append(mask_fwd)

                    # Save tmp mask
                    mask_fwd_vis = (255 * mask_fwd).astype(np.uint8)
                    imageio.imwrite(os.path.join(output_path_tmp, '%04d_%04d.png'%(i, j)), mask_fwd_vis)

            # Compute covisibility
            covis_masks = np.stack(covis_masks, axis=0)
            covis_mask = np.sum(covis_masks, axis=0)
            covis_mask = (covis_mask >= threshold).astype(np.float32)
            covis_mask = (255 * covis_mask).astype(np.uint8)

            imageio.imwrite(os.path.join(output_path, '%04d.png'%(i)), covis_mask)

            tbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path_src", type=str, help="Dataset path")
    parser.add_argument("--dataset_path_target", type=str, help="Dataset path")
    parser.add_argument("--model", help="restore RAFT checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    args = parser.parse_args()

    # Compute visibility of test images in train images
    input_path_target = os.path.join(args.dataset_path_target, "images_train")
    input_path_src = os.path.join(args.dataset_path_src, "images_test")
    output_path = os.path.join(args.dataset_path_src, "covis_mask")

    run(args, input_path_src, input_path_target, output_path)