import torch
import torch.nn as nn
import copy
from .position_encoding import VolumetricPositionEncoding as VolPE
from .geometry_attention import CorrespondenceAttentionLayer



class Outlier_Rejection(nn.Module):


    def __init__(self, config):
        super(Outlier_Rejection, self).__init__()

        self.num_layers = config['num_layers']
        self.pe_type = config["pe_type"]
        self.in_proj = nn.Linear( config['in_dim'], config['feature_dim'], bias=True)


        """pair-wise attention layers"""
        sa_layer = CorrespondenceAttentionLayer(config)
        self._6D_geometry_layers = nn.ModuleList()
        self.positional_encoding = VolPE(config)
        self.sigma_spat = config['sigma_spat']
        self.spatial_consistency_check = config['spatial_consistency_check']
        for i in range( self.num_layers):
            self._6D_geometry_layers.append(copy.deepcopy(sa_layer))


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


        feat = self.in_proj(corr_feat)


        for layer in self._6D_geometry_layers:
            feat = layer( feat, feat, pe_6d, pe_6d, data['vec_6d_mask'],data['vec_6d_mask'], compatibility = corr_compatibility )

        confidence = self.classification(feat).squeeze(-1)

        return confidence


    def _3D_to_6D(self, data):

        b_size=len(data['s_pcd'])
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



    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)