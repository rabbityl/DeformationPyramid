import torch
import torch.nn as nn
import copy
from .position_encoding import VolumetricPositionEncoding as VolPE
from .geometry_attention import CorrespondenceAttentionLayer
from .graph_attention import *




class Outlier_Rejection(nn.Module):


    def __init__(self, config):
        super(Outlier_Rejection, self).__init__()

        self.num_layers = config['num_layers']
        self.pe_type = config["pe_type"]
        self.in_proj = nn.Linear( config['in_dim'], config['feature_dim'], bias=True)

        self.alternate = config['alternate']

        """pair-wise attention layers"""
        sa_layer = CorrespondenceAttentionLayer(config)
        self._6D_geometry_layers = nn.ModuleList()
        self.positional_encoding = VolPE(config)
        self.sigma_spat = config['sigma_spat']
        self.spatial_consistency_check = config['spatial_consistency_check']
        for i in range( self.num_layers ):
            self._6D_geometry_layers.append(copy.deepcopy(sa_layer))


        """view graph cycle attention layers"""
        self.edge_search_radius = config['edge_R'] # search radius for edge
        cycle_attention_layer = GATLayer(config)
        self.graph_attention_layers = nn.ModuleList()
        for i in range( self.num_layers ):
            self.graph_attention_layers.append(copy.deepcopy(cycle_attention_layer))


        self.classification = nn.Sequential(
            nn.Linear(config['feature_dim'], 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True),
            nn.Sigmoid()
        )




    def forward(self, data):


        self._3D_to_6D( data )


        corr_feat = data['vec_6d']
        pos6d = data['vec_6d']

        n_frame, n_match, _ = pos6d.shape

        if self.spatial_consistency_check:
            with torch.no_grad():
                src_keypts, tgt_keypts = pos6d[...,:3], pos6d[...,3:]
                src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
                tgt_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
                corr_compatibility = src_dist - tgt_dist
                corr_compatibility = torch.clamp(1.0 - corr_compatibility ** 2 / self.sigma_spat ** 2, min=0)
        else:
            corr_compatibility = None


        if self.pe_type != 'none':
            pe_6d = self.positional_encoding(pos6d)
        else:
            pe_6d = None


        edge, edge_len = self.creat_graph_of_correspondences(data)

        feat = self.in_proj(corr_feat)

        if self.alternate == "geo" :

            for layer in self._6D_geometry_layers:
                feat = layer( feat, feat, pe_6d, pe_6d, data['vec_6d_mask'],data['vec_6d_mask'], compatibility = corr_compatibility )

        elif self.alternate == 'crs':

            feat = feat.view(-1, feat.shape[-1])
            for layer in self.graph_attention_layers:
                feat = layer( feat, edge, edge_len)
            feat = feat.view( n_frame, n_match, -1)

        elif self.alternate == "geocrs" :

            for layer in self._6D_geometry_layers:
                feat = layer( feat, feat, pe_6d, pe_6d, data['vec_6d_mask'],data['vec_6d_mask'], compatibility = corr_compatibility )
            feat = feat.view(-1, feat.shape[-1])
            for layer in self.graph_attention_layers:
                feat = layer( feat, edge, edge_len)
            feat = feat.view( n_frame, n_match, -1)

        elif self.alternate == "crsgeo" :

            feat = feat.view(-1, feat.shape[-1])
            for layer in self.graph_attention_layers:
                feat = layer( feat, edge, edge_len)
            feat = feat.view( n_frame, n_match, -1)
            for layer in self._6D_geometry_layers:
                feat = layer( feat, feat, pe_6d, pe_6d, data['vec_6d_mask'],data['vec_6d_mask'], compatibility = corr_compatibility )


        else :
            raise  KeyError( )


        confidence = self.classification(feat).squeeze(-1)

        return confidence

    # assert len(self._6D_geometry_layers) == len(self.graph_attention_layers)
    # for i in range(len(self._6D_geometry_layers)):
    #     feat = self._6D_geometry_layers[i](feat, feat, pe_6d, pe_6d, data['vec_6d_mask'], data['vec_6d_mask'],
    #                                        compatibility=corr_compatibility)
    #     feat = feat.view(-1, feat.shape[-1])
    #     feat = self.graph_attention_layers[i](feat, edge, edge_len)
    #     feat = feat.view(n_frame, n_match, -1)

    def _3D_to_6D(self, data):

        b_size=len(data['pcd_pairs'])
        ind = data['coarse_match_pred']
        bi, si, ti = ind[:, 0], ind[:, 1], ind[:, 2]
        vec6d_list = []
        batch_len = []

        batch_ind = []

        for i in range(b_size):
            bmask = bi == i
            batch_ind.append( torch.stack( [si[bmask], ti[bmask]], dim=-1 ))
            s_pos = data['s_pcd'][i][si[bmask]]
            t_pos = data['t_pcd'][i][ti[bmask]]
            batch_len.append(len(s_pos))
            vec_6d = torch.cat([s_pos, t_pos], dim=1)
            vec6d_list.append(vec_6d)

        lenth = max( batch_len )
        batch_vec6d = torch.zeros([b_size , lenth, 6]).type_as(vec6d_list[0])
        batch_mask = torch.zeros([b_size, lenth], dtype=torch.bool, device=batch_vec6d.device)
        batch_index = torch.zeros([b_size, lenth, 2], dtype=torch.long, device=batch_vec6d.device)

        for i in range(b_size):
            batch_vec6d[i][:batch_len[i]] = vec6d_list[i]
            batch_mask[i][:batch_len[i]] = 1
            batch_index[i][:batch_len[i]] = batch_ind[i]


        data['vec_6d'] = batch_vec6d
        data['vec_6d_mask'] = batch_mask
        data['vec_6d_ind'] = batch_index

    def creat_graph_of_correspondences(self, data):

        pcd_pairs = data['pcd_pairs']
        batch_vec6d = data['vec_6d']
        batch_mask = data['vec_6d_mask']

        n_pairs, n_match = batch_mask.shape

        n_pcd = pcd_pairs.max() + 1

        """ construct graph of correspondence"""
        #id of predicted matches
        corr_IDs = torch.arange(n_pairs * n_match, dtype=torch.long, device=batch_vec6d.device).reshape(n_pairs, n_match)

        edges = [ ]
        edge_lengths = []

        head_pos = batch_vec6d [..., :3]
        tail_pos = batch_vec6d [..., 3:]

        for p_i in range ( n_pcd ):  # iterate through point clouds

            is_head = pcd_pairs[:, 0] == p_i
            is_tail = pcd_pairs[:, 1] == p_i
            envolved = torch.logical_or (is_head , is_tail)

            if envolved.sum() <1 :
                continue

            is_head = is_head [envolved]
            is_tail = is_tail [envolved]
            head_pos_i = head_pos[ envolved ]
            tail_pos_i = tail_pos[ envolved ]
            corr_IDs_i = corr_IDs[ envolved ].view(-1)
            mask_i = batch_mask[envolved].view(-1)
            mask_i = (mask_i[:, None] * mask_i[None])

            pos_i = head_pos_i * is_head.float().view(-1,1,1) + tail_pos_i * is_tail.float().view(-1,1,1)
            pos_i = pos_i.view(-1, 3)

            # compute p2p distance
            dist_mat = torch.sum ( (pos_i[:, None] - pos_i[None])**2, dim=-1)
            valid_dist = dist_mat < self.edge_search_radius ** 2

            # # filter edge from save point clouds , but hold self-edge
            # A = torch.ones(len(is_head),n_match, n_match,  dtype=bool, device=dist_mat.device)
            # A = torch.block_diag(*A).fill_diagonal_(0)
            # valid_entry = ~A


            valid_edge = mask_i  * valid_dist


            edges_len = torch.sqrt( dist_mat[ valid_edge] )

            ind = torch.stack( [ corr_IDs_i[:, None].expand_as(valid_edge),
                                 corr_IDs_i[None].expand_as(valid_edge) ] , dim=-1)
            edge = ind[valid_edge]

            edges.append(edge)
            edge_lengths.append(edges_len)

        edges = torch.cat(edges,dim=0)
        edge_lengths = torch.cat(edge_lengths)

        return edges.T, edge_lengths


    def creat_graph(self, data):

        pcd_pairs = data['pcd_pairs']
        batch_vec6d = data['vec_6d']
        batch_mask = data['vec_6d_mask']

        n_pairs, n_match = batch_mask.shape

        n_pcd = pcd_pairs.max() + 1

        """ construct graph of correspondence"""
        # id of predicted matches
        corr_IDs = torch.arange(n_pairs * n_match, dtype=torch.long, device=batch_vec6d.device).reshape(n_pairs,
                                                                                                        n_match)

        edges = []
        edge_lengths = []

        head_pos = batch_vec6d[..., :3]
        tail_pos = batch_vec6d[..., 3:]

        for p_i in range(n_pcd):  # iterate through point clouds

            is_head = pcd_pairs[:, 0] == p_i
            is_tail = pcd_pairs[:, 1] == p_i
            envolved = torch.logical_or(is_head, is_tail)

            if envolved.sum() < 1:
                continue

            is_head = is_head[envolved]
            is_tail = is_tail[envolved]
            head_pos_i = head_pos[envolved]
            tail_pos_i = tail_pos[envolved]
            corr_IDs_i = corr_IDs[envolved].view(-1)
            mask_i = batch_mask[envolved].view(-1)
            mask_i = (mask_i[:, None] * mask_i[None])

            pos_i = head_pos_i * is_head.float().view(-1, 1, 1) + tail_pos_i * is_tail.float().view(-1, 1, 1)
            pos_i = pos_i.view(-1, 3)

            # compute p2p distance
            dist_mat = torch.sum((pos_i[:, None] - pos_i[None]) ** 2, dim=-1)
            valid_dist = dist_mat < self.edge_search_radius ** 2

            # filter edge from save point clouds , but hold self-edge
            A = torch.ones(len(is_head), n_match, n_match, dtype=bool, device=dist_mat.device)
            A = torch.block_diag(*A).fill_diagonal_(0)
            valid_entry = ~A

            valid_edge = mask_i * valid_entry * valid_dist

            edges_len = torch.sqrt(dist_mat[valid_edge])

            ind = torch.stack([corr_IDs_i[:, None].expand_as(valid_entry),
                               corr_IDs_i[None].expand_as(valid_entry)], dim=-1)
            edge = ind[valid_edge]

            edges.append(edge)
            edge_lengths.append(edges_len)

        edges = torch.cat(edges, dim=0)
        edge_lengths = torch.cat(edge_lengths)

        return edges.T, edge_lengths


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)