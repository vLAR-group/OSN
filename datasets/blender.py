import os
import os.path as osp
import json
import numpy as np
import imageio.v3 as imageio
from torch.utils.data import Dataset


class BlenderDataset(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 flow_window_size=1,
                 white_bkgd=False,
                 segm_sam=False):
        self.data_root = data_root
        self.split = split
        self.flow_window = list(range(-flow_window_size, 0)) + list(range(1, flow_window_size+1))
        self.white_bkgd = white_bkgd
        self.segm_sam = segm_sam

        # Load meta info
        with open(osp.join(data_root, 'transforms_%s.json'%(split)), 'r') as f:
            meta = json.load(f)

        camera_angle_x = float(meta['camera_angle_x'])
        self.img_h = float(meta['img_h'])
        self.img_w = float(meta['img_w'])
        self.focal = 0.5 * self.img_w / np.tan(0.5 * camera_angle_x)

        self.frames = meta['frames']
        self.n_sample = len(meta['frames'])

    def load_image(self, sid):
        file_path = osp.join(self.data_root, 'images_%s'%(self.split), '%04d.png'%(sid))
        img = imageio.imread(file_path)
        return img.astype(np.float32)

    def load_depth(self, sid):
        file_path = osp.join(self.data_root, 'depth_%s'%(self.split), '%04d.npy'%(sid))
        depth = np.load(file_path)
        return depth.astype(np.float32)

    def load_segm(self, sid):
        if self.split == 'train' and self.segm_sam:
            file_names = sorted(os.listdir(osp.join(self.data_root, 'segm_sam')))
            file_path = osp.join(self.data_root, 'segm_sam', file_names[sid])
        else:
            file_names = sorted(os.listdir(osp.join(self.data_root, 'segm_%s'%(self.split))))
            file_path = osp.join(self.data_root, 'segm_%s'%(self.split), file_names[sid])
        segm = np.load(file_path)
        return segm.astype(np.int32)

    def load_flow(self, sid1, sid2):
        file_path = osp.join(self.data_root, 'flow', '%04d_%04d.npy'%(sid1, sid2))
        flow = np.load(file_path)
        return flow.astype(np.float32)

    def load_pose(self, sid):
        frame_data = self.frames[sid]
        cam_pose = np.array(frame_data['transform_matrix'])
        obj_pose = np.array(frame_data['object_poses'])
        pose = np.concatenate([np.expand_dims(cam_pose, 0), obj_pose], 0)
        return pose.astype(np.float32)

    def load_all_images(self):
        imgs = []
        for sid in range(self.n_sample):
            img = self.load_image(sid)
            imgs.append(img)
        imgs = np.stack(imgs, 0)
        imgs = (imgs / 255.).astype(np.float32)
        if self.white_bkgd:
            imgs = imgs[..., :3] * imgs[..., 3:] + (1. - imgs[..., 3:])
        else:
            imgs = imgs[..., :3]
        return imgs

    def load_all_depths(self):
        depths = []
        for sid in range(self.n_sample):
            depth = self.load_depth(sid)
            depths.append(depth)
        return np.stack(depths, 0)

    def load_all_segms(self):
        segms = []
        for sid in range(self.n_sample):
            segm = self.load_segm(sid)
            segms.append(segm)
        return np.stack(segms, 0)

    def load_all_poses(self):
        poses = []
        for sid in range(self.n_sample):
            pose = self.load_pose(sid)
            poses.append(pose)
        return np.stack(poses, 0)

    def load_neighbor_flows(self):
        flows, sid2_list = [], []
        for sid in range(self.n_sample):
            # Sample a neighboring frame for optical flow
            offsets = [offset for offset in self.flow_window if (sid+offset > -1) and (sid+offset) < self.n_sample]
            offset = np.random.choice(offsets)
            sid2 = sid + offset

            flow = self.load_flow(sid, sid2)
            flows.append(flow)
            sid2_list.append(sid2)
        return np.stack(flows, 0), sid2_list


print('Good')