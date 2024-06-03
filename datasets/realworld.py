import os
import os.path as osp
import json
import numpy as np
import imageio.v3 as imageio
from torch.utils.data import Dataset


class RealWorldDataset(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 segm_sam=False):
        self.data_root = data_root
        self.split = split
        self.segm_sam = segm_sam

        # Load meta info
        with open(osp.join(data_root, 'transforms_%s.json' % (split)), 'r') as f:
            meta = json.load(f)

        self.img_h = float(meta['img_h'])
        self.img_w = float(meta['img_w'])
        self.focal = float(meta['focal'])

        self.frames = meta['frames']
        self.n_sample = len(meta['frames'])

    def load_image(self, sid):
        file_path = osp.join(self.data_root, 'images_%s' % (self.split), '%04d.png' % (sid))
        img = imageio.imread(file_path)
        return img.astype(np.float32)

    def load_segm(self, sid):
        assert self.split == 'train', 'No segmentation for test set!'
        if self.segm_sam:
            file_names = sorted(os.listdir(osp.join(self.data_root, 'segm_sam')))
            file_path = osp.join(self.data_root, 'segm_sam', file_names[sid])
        else:
            file_names = sorted(os.listdir(osp.join(self.data_root, 'segm_%s'%(self.split))))
            file_path = osp.join(self.data_root, 'segm_%s'%(self.split), file_names[sid])
        segm = np.load(file_path)
        return segm.astype(np.int32)

    def load_pose(self, sid):
        frame_data = self.frames[sid]
        pose = np.array(frame_data['transform_matrix'])
        return pose.astype(np.float32)

    def load_all_images(self):
        imgs = []
        for sid in range(self.n_sample):
            img = self.load_image(sid)
            imgs.append(img)
        imgs = np.stack(imgs, 0)
        imgs = (imgs / 255.).astype(np.float32)
        return imgs

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