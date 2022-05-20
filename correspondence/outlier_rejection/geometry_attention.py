
import torch
import torch.nn as nn
import copy
from .position_encoding import VolumetricPositionEncoding as VolPE



class CorrespondenceAttentionLayer(nn.Module):

    def __init__(self, config):

        super(CorrespondenceAttentionLayer, self).__init__()

        d_model = config['feature_dim']
        nhead =  config['n_head']

        self.dim = d_model // nhead
        self.nhead = nhead
        self.pe_type = config['pe_type']
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None, compatibility = None):

        bs = x.size(0)
        q, k, v = x, source, source
        qp, kvp  = x_pe, source_pe
        q_mask, kv_mask = x_mask, source_mask

        if self.pe_type == 'sinusoidal':
            #w(x+p), attention is all you need : https://arxiv.org/abs/1706.03762
            if qp is not None: # disentangeld
                q = q + qp
                k = k + kvp
            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'rotary':
            #Rwx roformer : https://arxiv.org/abs/2104.09864

            qw = self.q_proj(q)
            kw = self.k_proj(k)
            vw = self.v_proj(v)

            if qp is not None: # disentangeld
                q_cos, q_sin = qp[...,0] ,qp[...,1]
                k_cos, k_sin = kvp[...,0],kvp[...,1]
                qw = VolPE.embed_rotary(qw, q_cos, q_sin)
                kw = VolPE.embed_rotary(kw, k_cos, k_sin)

            qw = qw.view(bs, -1, self.nhead, self.dim)
            kw = kw.view(bs, -1, self.nhead, self.dim)
            vw = vw.view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'none':

            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)



        else:
            raise KeyError()

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)

        if compatibility is not None:
            a = a * compatibility[..., None]

        if kv_mask is not None:
            a.masked_fill_( q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))



        a =  a / qw.size(3) **0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]
        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        e = x + message

        return e




class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels//2, kernel_size=1),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels//2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

        # for block in [ self.projection_q, self.fc_message ]:
        #
        #     in_param4 = sum(p.numel() for p in block.parameters() if p.requires_grad)
        #     print( in_param4 )

    def forward(self, feat):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        # combine the feature similarity with spatial consistency
        # weight = torch.softmax(attention[:, None, :, :] * feat_attention, dim=-1)
        weight = torch.softmax(  feat_attention, dim=-1) #spatial consistency is removed due to non-rigidity
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])
        message = self.fc_message(message)
        res = feat + message
        return res


class NonLocalNet(nn.Module):
    def __init__(self, in_dim=6, num_layers=6, num_channels=128):
        super(NonLocalNet, self).__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleDict()
        self.layer0 = nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True),
                # nn.InstanceNorm1d(num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True)
            )
            self.blocks[f'PointCN_layer_{i}'] = layer
            self.blocks[f'NonLocal_layer_{i}'] = NonLocalBlock(num_channels)


        self.classification = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, corr_feat):
        """
        Input:
            - corr_feat:          [bs, in_dim, num_corr]   input feature map

        Output:
            - feat:               [bs, num_channels, num_corr] updated feature
        """
        corr_feat=torch.permute(corr_feat, (0,2,1))
        feat = self.layer0(corr_feat)
        for i in range(self.num_layers):
            feat = self.blocks[f'PointCN_layer_{i}'](feat)
            feat = self.blocks[f'NonLocal_layer_{i}'](feat)

        confidence = self.classification(feat ).squeeze(1)
        return confidence