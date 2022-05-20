import os, sys, glob, torch
sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, KDTree_corr
from lib.utils import load_obj
HMN_intrin = np.array( [443, 256, 443, 250 ])
cam_intrin = np.array( [443, 256, 443, 250 ])

from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences



class _4DMatch_Multiview(Dataset):

    def __init__(self, config, split, data_augmentation=True):
        super(_4DMatch_Multiview, self).__init__()

        assert split in ['train','val','test']

        if 'overfit' in config.exp_dir:
            d_slice = config.batch_size
        else :
            d_slice = None

        self.entries = self.read_entries(  config.split[split] , config.data_root, d_slice=d_slice )

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

        self.overlap_radius = 0.0375

        self.cache = {}
        self.cache_size = 30000

        self.overlap_threshold = 0.15


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


        if index in self.cache:
            entry = self.cache[index]

        else :
            entry = np.load(self.entries[index],allow_pickle=True)
            if len(self.cache) < self.cache_size:
                self.cache[index] = entry

        # """save data"""
        # np.savez_compressed(fname,
        #                     pcds=pcd_sampled,
        #                     pcd_pairs=edges,
        #                     pairwise_flows=pairwise_flows,
        #                     pairwise_overlap=edge_info,
        #                     _2axis_flow=_2Axisflow_sampled,
        #                     axis_node=axis_node,
        #                     poses =Transforms
        #                     )

        pcds = entry['pcds']
        pcd_pairs = entry ['pcd_pairs']
        pairwise_flows = entry ['pairwise_flows']
        pairwise_overlap = entry ['pairwise_overlap']
        _2axis_flow = entry['_2axis_flow'] # [edges, edge_info]
        axis_node = entry['axis_node']
        poses = entry['poses']

        '''filter pairs with overlap threshold'''
        valid_pairs = np.logical_and(
            pairwise_overlap[:,0] > self.overlap_threshold,
            pairwise_overlap[:,1] > self.overlap_threshold )
        pcd_pairs = pcd_pairs [valid_pairs]
        pairwise_flows = pairwise_flows [ valid_pairs]
        pairwise_overlap = pairwise_overlap [valid_pairs]

        #R * ( Ps + flow ) + t  = Pt
        return pcds, pcd_pairs, pairwise_flows,pairwise_overlap, _2axis_flow, axis_node, poses


