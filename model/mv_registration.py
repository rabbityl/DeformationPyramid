import torch
import os, sys
from skimage import io
from .geometry import *
import pytorch3d
import torch.optim as optim

from geomloss import SamplesLoss

#cpd
from pycpd import DeformableRegistration as CPD_Deform

from .geometry import map_pixel_to_pcd,pc_2_uv
from .loss import silhouette_cost, arap_cost, landmark_cost, compute_truncated_chamfer_distance, chamfer_dist, nerfies_regularization
from .point_render import PCDRender
from .nets import *
import os

sys.path.append("../")
from utils.vis import visualize_pcds_list


BCE = nn.BCELoss()

import copy

class MVRegistration():


    def __init__(self, config):


        self.tgt_pcd = None
        self.src_pcd = None
        self.device = config.device
        self.config = config
        self.deformation_model = config.deformation_model




    def load_raw_pcds_from_depth(self, source_depth_path, tgt_depth_path, K, landmarks=None):
        ''' creat deformation graph for N-ICP based registration
        '''

        assert self.deformation_model == "ED"

        self.intrinsics = K

        """initialize deformation graph"""
        depth_image = io.imread(source_depth_path)
        image_size = (depth_image.shape[0], depth_image.shape[1])
        data = get_deformation_graph_from_depthmap( depth_image, K, self.config)
        self.graph_nodes = data['graph_nodes'].to(self.device)
        self.graph_edges = data['graph_edges'].to(self.device)
        self.graph_edges_weights = data['graph_edges_weights'].to(self.device)
        # self.graph_clusters = data['graph_clusters']


        """initialize point clouds"""
        valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        self.src_pcd_raw = data["point_image"][valid_pixels].to(self.device)
        self.point_anchors = data["pixel_anchors"][valid_pixels].long().to(self.device)
        self.anchor_weight = data["pixel_weights"][valid_pixels].to(self.device)
        self.anchor_loc = data["graph_nodes"][self.point_anchors].to(self.device)
        self.frame_point_len = [len(self.src_pcd_raw)]


        """pixel to pcd map"""
        self.src_pix_2_pcd_map = [map_pixel_to_pcd(valid_pixels)]


        """define pcd renderer"""
        # self.renderer = PCDRender(K, img_size=image_size)


        """load target frame"""
        tgt_depth = io.imread( tgt_depth_path )/1000.
        depth_mask = torch.from_numpy(tgt_depth > 0)
        tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
        self.tgt_pcd_raw = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
        self.tgt_pix_2_pcd_map = map_pixel_to_pcd(depth_mask)

        if landmarks is not None:
            s_uv , t_uv = landmarks
            s_id = self.src_pix_2_pcd_map[-1][s_uv[:, 1], s_uv[:, 0]]
            t_id = self.tgt_pix_2_pcd_map [ t_uv[:,1], t_uv[:,0]]
            valid_id = (s_id>-1) * (t_id>-1)
            s_ldmk = s_id[valid_id]
            t_ldmk = t_id[valid_id]
            self.landmarks = (s_ldmk, t_ldmk)
        else:
            self.landmarks = None



    def register(self,  pcd_list, landmarks_dict, pcd_pair,  static=0, debug=False, timer = None):

        config = self.config
        max_break_count=config.max_break_count
        break_threshold_ratio=config.break_threshold_ratio


        means = []

        # load pcd
        for i in range( len(pcd_list)):
            pcd_list[i] = pcd_list[i].to(self.device)
            mean = pcd_list[i].mean(dim=0, keepdims=True)
            pcd_list[i] = pcd_list[i] - mean
            means.append(mean)

        for pair in pcd_pair:

            sid, tid = pair

            landmarks_dict [ (sid, tid) ]['s'] -= means[ sid ]
            landmarks_dict [ (sid, tid) ]['t'] -= means[ tid ]


        NDPs = { }



        for i in range ( len(pcd_list)) :

            if i != static :

                NDPs[i] = Deformation_Pyramid(  depth=config.depth,
                                                width=config.width,
                                                device=self.device,
                                                k0=config.k0,
                                                m=config.m,
                                                nonrigidity_est=config.w_reg > 0,
                                                rotation_format=config.rotation_format,
                                                motion=config.motion_type)




        if debug:
            visualize_pcds_list( [ v for k,v, in pcd_list.items() ] )


        level_wise=[ [  v.detach().cpu().numpy()   for k,v, in pcd_list.items() ]]

        for level in range ( config.m ):

            """freeze non-optimized level"""
            params = []
            for k, ndp in NDPs.items():
                ndp.gradient_setup(optimized_level=level)
                params = params + list( ndp.pyramid[level].parameters())


            optimizer = optim.Adam(  params, lr= self.config.lr )


            break_counter = 0
            loss_prev = 1e+6


            landmarks_warped = copy.deepcopy( landmarks_dict )


            """optimize current level"""
            for iter in range(self.config.iters):

                # warp ldmk

                loss = 0

                n_ldmk = 0

                for k, v in landmarks_dict.items():

                    sid, tid= k

                    src_ldmk = v['s']
                    tgt_ldmk = v['t']

                    src_ldmk_warped = landmarks_warped[k]['s']
                    tgt_ldmk_warped = landmarks_warped[k]['t']


                    if sid !=static:

                        warped_src, _ = NDPs[sid].warp( src_ldmk, max_level=level, min_level=level )
                        loss_s2t =  torch.sum((warped_src - tgt_ldmk_warped) ** 2, dim=-1)
                        loss = loss + loss_s2t.sum()
                        n_ldmk += src_ldmk.shape[0]
                        landmarks_warped[k]['s'] = warped_src.detach()


                    if tid !=static:

                        warped_tgt, _ = NDPs[tid].warp( tgt_ldmk, max_level=level, min_level=level )
                        loss_t2s =  torch.sum((warped_tgt - src_ldmk_warped) ** 2, dim=-1)
                        loss = loss + loss_t2s.sum()
                        n_ldmk += tgt_ldmk.shape[0]
                        landmarks_warped[k]['t'] = warped_tgt.detach()


                loss = loss / n_ldmk

                # early stop
                if loss.item() < 1e-4:
                    break
                if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                    break_counter += 1
                if break_counter >= max_break_count:
                    break
                loss_prev = loss.item()

                if timer: timer.tic("backprop")
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if timer: timer.toc("backprop")

                if debug:
                    pass
                    # print(iter, loss)

            if debug:
                # for k, v in landmarks_dict.items():
                #     sid, tid= k
                #     if sid !=static :
                #         warped_src, _ = NDPs[sid].warp( pcd_list[sid], max_level=level )
                #     if tid !=static  :
                #         warped_tgt, _ = NDPs[tid].warp( pcd_list[tid], max_level=level )
                #     visualize_pcds(  tgt_pcd= warped_tgt, warped_pcd = warped_src)

                print( 'level', level)

                warped_list = []
                for i in range(len(pcd_list)):
                    if i != static:
                        warped, _ = NDPs[i].warp(pcd_list[i], max_level=level)
                        warped_list.append(warped.detach().cpu().numpy())
                    else:
                        warped_list.append(pcd_list[i].cpu().numpy())

                visualize_pcds_list(warped_list)

                level_wise.append(warped_list)



            # use warped points for next level
            landmarks_dict = landmarks_warped

        # if debug:
        #     # for k, v in landmarks_dict.items():
        #     #     sid, tid= k
        #     #     if sid !=static :
        #     #         warped_src, _ = NDPs[sid].warp( pcd_list[sid], max_level=level )
        #     #     if tid !=static  :
        #     #         warped_tgt, _ = NDPs[tid].warp( pcd_list[tid], max_level=level )
        #     #     visualize_pcds(  tgt_pcd= warped_tgt, warped_pcd = warped_src)
        #
        #     warped_list = []
        #     for i in range(len(pcd_list)):
        #         if i != static:
        #             warped, _ = NDPs[i].warp(pcd_list[i], max_level=level)
        #             warped_list.append( warped)
        #         else :
        #             warped_list.append( pcd_list[i] )
        #
        #     visualize_pcds_list(warped_list)


        # return warped_pcd,  iter_cnt, timer

        return level_wise


    def gather_ldmks(self,  landmarks_dict, max):
        pass

        anchors = { i:[] for i in range(max) }

        for k, v in landmarks_dict.items() :

            s, t  = k

            anchors[s].append( v['s'] )

            anchors[t].append( v['t'] )

