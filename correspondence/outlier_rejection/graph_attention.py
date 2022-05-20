import torch
import torch.nn as nn



class GAT(torch.nn.Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                d_in = num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                d_out = num_features_per_layer[i+1],
                n_head =num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, data):
        return self.gat_net(data)



class GATLayer (torch.nn.Module):


    def __init__(self, config) :
        super().__init__()

        # d_in, d_out, n_head, concat=True, activation=nn.ELU(), dropout_prob=0.6, add_skip_connection=True, bias=True):

        d_in = config['feature_dim']
        n_head = config['n_head']

        assert d_in%n_head ==0
        d_out = d_in//n_head

        dropout_prob = config['dropout']
        concat = True
        activation = nn.ELU()
        add_skip_connection = True
        bias = True



        self.n_head = n_head
        self.d_out = d_out
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip = add_skip_connection


        # You can treat this one matrix as num_of_heads independent W matrices
        self.input_proj = nn.Linear(d_in, n_head * d_out, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, n_head, d_out))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, n_head, d_out))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(n_head * d_out))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(d_out))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(d_in, n_head * d_out, bias=False)
        else:
            self.register_parameter('skip_proj', None)


        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)


        self.init_params()


    def init_params(self):

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, in_feat, edge_ind, edge_len):

        n_nodes = in_feat.shape[0]
        assert edge_ind.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_ind.shape}'

        in_feat = self.dropout(in_feat)


        proj_feat = self.input_proj(in_feat).view(-1, self.n_head, self.d_out)
        proj_feat = self.dropout(proj_feat)  # in the official GAT imp they did dropout here as well


        """scoring feature"""
        src_score = (proj_feat * self.scoring_fn_source).sum(dim=-1)
        tgt_score = (proj_feat * self.scoring_fn_target).sum(dim=-1)


        """broadcast feature"""
        src_ind, tgt_ind = edge_ind[0] , edge_ind[1]
        src_score = src_score[ src_ind ]
        tgt_score = tgt_score[ tgt_ind ]
        src_feat = proj_feat [ src_ind ]
        edge_score = self.leakyReLU(src_score + tgt_score)


        """compute per-edge attention"""
        edge_score = edge_score - edge_score.max()
        edge_score = edge_score.exp()
        tgt_ind_broadcasted = tgt_ind.unsqueeze(-1).expand_as ( edge_score )
        neighborhood_sums =  torch.zeros([ n_nodes, edge_score.shape[1] ], dtype=edge_score.dtype, device=edge_score.device)
        neighborhood_sums.scatter_add_(0, tgt_ind_broadcasted , edge_score)
        neighborhood_sums = neighborhood_sums[ tgt_ind ]
        att_per_edge = edge_score / (neighborhood_sums + 1e-16)


        """attention dropout"""
        att_per_edge = self.dropout(att_per_edge)


        """Neighborhood aggregation"""
        src_feat_weighted = src_feat * att_per_edge[..., None]
        out_feat =  torch.zeros([ n_nodes, src_feat_weighted.shape[1], src_feat_weighted.shape[2] ], dtype=src_feat_weighted.dtype, device=src_feat_weighted.device)
        tgt_index_broadcasted =  tgt_ind[...,None,None].expand_as( src_feat_weighted)
        out_feat.scatter_add_(0, tgt_index_broadcasted, src_feat_weighted)


        """output feature"""
        out_nodes_features = out_feat.view(n_nodes, -1)
        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features