import os, sys, glob, torch
sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])


class _4DMatch(Dataset):

    def __init__(self, config, split, data_augmentation=True):
        super(_4DMatch, self).__init__()

        assert split in ['train','val','test']


        self.entries = self.read_entries(  config.split[split] , config.data_root, d_slice=None )

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

        self.overlap_radius = 0.0375

        # self.cache = {}
        # self.cache_size = 30000

        self.load_raw_depth = config.use_depth
        if config.use_depth:
            self.img_root=  config.img_root



    def read_entries (self, split, data_root, d_slice=None, shuffle= False):
        entries = glob.glob(os.path.join(data_root, split, "*/*.npz"))
        if shuffle:
            random.shuffle(entries)
        if d_slice:
            return entries[:d_slice]
        return entries


    def __len__(self):
        return len(self.entries )


    def __getitem__(self, index, debug=False):



        with np.load(self.entries[index]) as entry :

            # get transformation
            rot = entry['rot']
            trans = entry['trans']
            s2t_flow = entry['s2t_flow']
            src_pcd = entry['s_pc']
            tgt_pcd = entry['t_pc']
            correspondences = entry['correspondences'] # obtained with search radius 0.015 m
            src_pcd_deformed = src_pcd + s2t_flow
            if "metric_index" in entry:
                metric_index = entry['metric_index'].squeeze()
            else:
                metric_index = None

        if self.load_raw_depth:
            entry_name = self.entries[index]
            # entry_key = entry_name.split("/")[-1]
            entry_info = entry_name.split("/")[-1].split("_")
            entry_info[-1] = entry_info[-1][:-4]
            scam, sid, tcam, tid = entry_info
            seq_name = entry_name.split("/")[-2]
            seq_raw = os.path.join(self.img_root, seq_name )
            s_depth = os.path.join( seq_raw, "depth", scam + "_" + sid + ".png")
            t_depth = os.path.join( seq_raw, "depth", tcam + "_" + tid + ".png")
            depth_paths = [s_depth ,t_depth]
            cam_intrin = np.loadtxt ( os.path.join( seq_raw , "cam1intr.txt"))
        else :
            depth_paths = None
            cam_intrin = None


        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]


        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.show()



        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                src_pcd_deformed = np.matmul(rot_ab, src_pcd_deformed.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
            s2t_flow = src_pcd_deformed - src_pcd


        if debug:
            # wrapp_src = (np.matmul(rot, src_pcd.T)+ trans).T
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_red)
            mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
            mlab.show()


        if (trans.ndim == 1):
            trans = trans[:, None]


        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)


        #R * ( Ps + flow ) + t  = Pt
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index, depth_paths, cam_intrin



