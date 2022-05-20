import torch
import torch.nn as nn
BCE = nn.BCELoss()
import open3d as o3d

import torch.optim as optim

import  yaml
from easydict import EasyDict as edict

from utils.benchmark_utils import setup_seed

import numpy as np

from model.nets import Deformation_Pyramid
from model.loss import compute_truncated_chamfer_distance
import argparse


setup_seed(0)




if __name__ == "__main__":

    config = {
        "gpu_mode": True,

        "iters": 500,
        "lr": 0.01,
        "max_break_count": 15,
        "break_threshold_ratio": 0.001,

        "samples": 6000,

        "motion_type": "Sim3",
        "rotation_format": "euler",

        "m": 9,
        "k0": -8,
        "depth": 3,
        "width": 128,
        "act_fn": "relu",

        "w_reg": 0,
        "w_ldmk": 0,
        "w_cd": 0.1
    }

    config = edict(config)

    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help= 'Path to the src mesh.')
    parser.add_argument('-t', type=str, help='Path to the tgt mesh.')
    args = parser.parse_args()


    S=args.s
    T=args.t

    """read S, sample pts"""
    src_mesh = o3d.io.read_triangle_mesh( S )
    src_mesh.compute_vertex_normals()
    pcd1 =  src_mesh.sample_points_uniformly(number_of_points=config.samples)
    pcd1.paint_uniform_color([0, 0.706, 1])
    src_pcd = np.asarray(pcd1.points, dtype=np.float32)

    o3d.visualization.draw_geometries([src_mesh])

    """read T, sample pts"""
    tgt_mesh = o3d.io.read_triangle_mesh( T )
    tgt_mesh.compute_vertex_normals()
    pcd2 =  tgt_mesh.sample_points_uniformly(number_of_points=config.samples)
    tgt_pcd = np.asarray(pcd2.points, dtype=np.float32)

    o3d.visualization.draw_geometries([tgt_mesh])


    """load data"""
    src_pcd, tgt_pcd = map( lambda x: torch.from_numpy(x).to(config.device), [src_pcd, tgt_pcd ] )



    """construct model"""
    NDP = Deformation_Pyramid(depth=config.depth,
                              width=config.width,
                              device=config.device,
                              k0=config.k0,
                              m=config.m,
                              nonrigidity_est=config.w_reg > 0,
                              rotation_format=config.rotation_format,
                              motion=config.motion_type)



    """cancel global translation"""
    src_mean = src_pcd.mean(dim=0, keepdims=True)
    tgt_mean = tgt_pcd.mean(dim=0, keepdims=True)
    src_pcd = src_pcd - src_mean
    tgt_pcd = tgt_pcd - tgt_mean




    s_sample = src_pcd
    t_sample = tgt_pcd


    for level in range(NDP.n_hierarchy):

        """freeze non-optimized level"""
        NDP.gradient_setup(optimized_level=level)

        optimizer = optim.Adam(NDP.pyramid[level].parameters(), lr=config.lr)

        break_counter = 0
        loss_prev = 1e+6

        """optimize current level"""
        for iter in range(config.iters):


            s_sample_warped, data = NDP.warp(s_sample, max_level=level, min_level=level)

            loss = compute_truncated_chamfer_distance(s_sample_warped[None], t_sample[None], trunc=1e+9)


            if level > 0 and config.w_reg > 0:
                nonrigidity = data[level][1]
                target = torch.zeros_like(nonrigidity)
                reg_loss = BCE(nonrigidity, target)
                loss = loss + config.w_reg * reg_loss


            # early stop
            if loss.item() < 1e-4:
                break
            if abs(loss_prev - loss.item()) < loss_prev * config.break_threshold_ratio:
                break_counter += 1
            if break_counter >= config.max_break_count:
                break
            loss_prev = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # use warped points for next level
        s_sample = s_sample_warped.detach()



    """warp-original mesh verttices"""
    NDP.gradient_setup(optimized_level=-1)
    mesh_vert = torch.from_numpy(np.asarray(src_mesh.vertices, dtype=np.float32)).to(config.device)
    mesh_vert = mesh_vert - src_mean
    warped_vert, data = NDP.warp(mesh_vert)
    warped_vert = warped_vert.detach().cpu().numpy()
    src_mesh.vertices = o3d.utility.Vector3dVector(warped_vert)
    o3d.visualization.draw_geometries([src_mesh])

    """dump results"""
    # o3d.io.write_triangle_mesh("sim3_demo/things4D/" + sname + "-fit.ply", src_mesh)