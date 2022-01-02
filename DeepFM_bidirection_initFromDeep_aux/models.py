import torch
import torch.nn as nn
from layers import (SparseEmbedding, Wide,
                    MultiLayerPerceptron, factorization_machine,
                    RecurrentLink, RecurrentCross)


class WideDeepRecurrentRecurrent(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        features_to_use = configs.features_to_use
        dense_field_dims = configs.dense_field_dims
        sparse_field_dims = configs.sparse_field_dims
        # sparse_cross_field_dims = configs.sparse_cross_field_dims
        emb_dim = configs.emb_dim
        wide_config = configs.wide
        deep_config = configs.deep
        rnn_config = configs.rnn
        self.use_seq_link = rnn_config['use_seq_link']
        self.use_seq_cross = rnn_config['use_seq_cross']
        self.init_from_deep = rnn_config['init_from_deep']
        self.norm = nn.BatchNorm1d(dense_field_dims)
        self.sparse_emb = SparseEmbedding(sparse_field_dims, emb_dim)
        # if configs.dense_emb:
        #     self.dense_emb = DenseEmbedding(dense_field_dims, emb_dim)
        #     dense_field_dims *= emb_dim

        self.wide = Wide(dense_field_dims, sparse_field_dims, wide_config['output_dim'])
        deep_input_dim = dense_field_dims + len(sparse_field_dims) * emb_dim
        self.deep = MultiLayerPerceptron(input_dim=deep_input_dim,
                                         hidden_dims=deep_config['hidden_dims'],
                                         output_dim=deep_config['output_dim'],
                                         batchnorm=deep_config['batchnorm'],
                                         dropout=deep_config['dropout'])
        head_input_dim = wide_config['output_dim'] + deep_config['output_dim']

        if self.use_seq_link:
            self.rnn_link = RecurrentLink(rnn_config['hidden_dim'], rnn_config['num_layers'],
                                          rnn_config['bidirectional'], deep_config['dropout'],
                                          rnn_config['type'], rnn_config['init_from_deep'],
                                          features_to_use['seq_link']['dense'],
                                          features_to_use['seq_link']['sparse'],
                                          configs.sparse_num_emb, emb_dim,
                                          rnn_config['auxiliary_loss'])
            self.rnn_type = rnn_config['type']
            head_input_dim += rnn_config['output_dim']
            if self.init_from_deep:
                self.linear_deep_to_rnn = nn.Linear(deep_input_dim, rnn_config['hidden_dim'])
                self.num_states = rnn_config['num_layers'] * (2 if rnn_config['bidirectional'] else 1)
        if self.use_seq_cross:
            self.rnn_cross = RecurrentCross(len(features_to_use['seq_cross']['dense']),
                                            rnn_config['hidden_dim'], rnn_config['num_layers'],
                                            rnn_config['bidirectional'], deep_config['dropout'],
                                            rnn_config['type'], rnn_config['init_from_deep'])
            head_input_dim += rnn_config['output_dim']

        head_hidden_dim = configs.head['head_hidden_dim']
        self.head = nn.Sequential(nn.Linear(head_input_dim, head_hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(head_hidden_dim, 1))

    def forward(self, dense, sparse, sparse_cross,
                seq_link_dense, seq_link_sparse, num_links,
                seq_cross_dense, num_crosses):
        '''
        Args:
            dense: (N, num_dense_features)
            sparse: (N, len(sparse_field_dims))
            sparse_cross: (N, len(sparse_cross_field_dims))
            seq_link_dense: (N, T1, d1)
            seq_link_sparse: (N, T1, d2)
            seq_cross_dense: (N, T2, d3)
        Output:
            eta: (N, 1)
            arrival_pred: Optional[(sum(num_links), 5)]
        '''

        # if self.configs.dense_emb:
        #     dense = self.dense_emb(dense)
        dense = self.norm(dense)
        out_wide = self.wide(dense, sparse)
        deep_rnn_shared = torch.cat([dense, self.sparse_emb(sparse).view(dense.size(0), -1)], dim=-1)
        out_deep = self.deep(deep_rnn_shared)

        if self.use_seq_link:
            if self.init_from_deep:
                deep_rnn_shared = deep_rnn_shared.detach()[None].repeat(self.num_states, 1, 1)
                deep_rnn_shared = self.linear_deep_to_rnn(deep_rnn_shared)
                initial_state = (deep_rnn_shared, deep_rnn_shared) if self.rnn_type == "LSTM" else deep_rnn_shared
            else:
                initial_state = None
            ht_link, arrival_pred = self.rnn_link(seq_link_dense, seq_link_sparse, num_links, initial_state)
            if self.use_seq_cross:
                ht_cross = self.rnn_cross(seq_cross_dense, num_crosses, initial_state)
                head_input = torch.cat([out_wide, out_deep, ht_link, ht_cross], dim=-1)
            else:
                head_input = torch.cat([out_wide, out_deep, ht_link], dim=-1)
        else:
            head_input = torch.cat([out_wide, out_deep], dim=-1)
            arrival_pred = None

        return self.head(head_input), arrival_pred


class DeepFMRecurrentRecurrent(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        features_to_use = configs.features_to_use
        dense_field_dims = configs.dense_field_dims
        sparse_field_dims = configs.sparse_field_dims
        emb_dim = configs.emb_dim
        deep_config = configs.deep
        rnn_config = configs.rnn
        self.use_seq_link = rnn_config['use_seq_link']
        self.use_seq_cross = rnn_config['use_seq_cross']
        self.init_from_deep = rnn_config['init_from_deep']
        self.norm = nn.BatchNorm1d(dense_field_dims)
        self.sparse_emb = SparseEmbedding(sparse_field_dims, emb_dim)
        self.dense_emb = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(dense_field_dims)])
        dense_field_dims *= emb_dim

        deep_input_dim = dense_field_dims + len(sparse_field_dims) * emb_dim
        self.deep = MultiLayerPerceptron(input_dim=deep_input_dim,
                                         hidden_dims=deep_config['hidden_dims'],
                                         output_dim=deep_config['output_dim'],
                                         batchnorm=deep_config['batchnorm'],
                                         dropout=deep_config['dropout'])
        head_input_dim = 1 + deep_config['output_dim']

        if self.use_seq_link:
            self.rnn_link = RecurrentLink(rnn_config['hidden_dim'], rnn_config['num_layers'],
                                          rnn_config['bidirectional'], deep_config['dropout'],
                                          rnn_config['type'], rnn_config['init_from_deep'],
                                          features_to_use['seq_link']['dense'],
                                          features_to_use['seq_link']['sparse'],
                                          configs.sparse_num_emb, emb_dim,
                                          rnn_config['auxiliary_loss'])
            self.rnn_type = rnn_config['type']
            head_input_dim += rnn_config['output_dim']
            if self.init_from_deep:
                self.linear_deep_to_rnn = nn.Linear(deep_input_dim, rnn_config['hidden_dim'])
                self.num_states = rnn_config['num_layers'] * (2 if rnn_config['bidirectional'] else 1)
        if self.use_seq_cross:
            self.rnn_cross = RecurrentCross(len(features_to_use['seq_cross']['dense']),
                                            rnn_config['hidden_dim'], rnn_config['num_layers'],
                                            rnn_config['bidirectional'], deep_config['dropout'],
                                            rnn_config['type'], rnn_config['init_from_deep'])
            head_input_dim += rnn_config['output_dim']

        head_hidden_dim = configs.head['head_hidden_dim']
        self.head = nn.Sequential(nn.Linear(head_input_dim, head_hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(head_hidden_dim, 1))

    def forward(self, dense, sparse, sparse_cross,
                seq_link_dense, seq_link_sparse, num_links,
                seq_cross_dense, num_crosses):
        '''
        Args:
            dense: (N, num_dense_features)
            sparse: (N, len(sparse_field_dims))
            sparse_cross: we did not use sparse_cross in deepfm
            seq_link_dense: (N, T1, d1)
            seq_link_sparse: (N, T1, d2)
            seq_cross_dense: (N, T2, d3)
        Output:
            eta: (N, 1)
            arrival_pred: Optional[(sum(num_links), 5)]
        '''
        N = dense.size(0)
        dense = self.norm(dense)
        dense = torch.stack([emb(dense[..., i:(i+1)]) for i, emb in enumerate(self.dense_emb)], dim=1)
        sparse = self.sparse_emb(sparse)
        input_fm = torch.cat([dense, sparse], dim=1)
        out_fm = factorization_machine(input_fm)
        deep_rnn_shared = input_fm.view(N, -1)
        out_deep = self.deep(deep_rnn_shared)

        if self.use_seq_link:
            if self.init_from_deep:
                deep_rnn_shared = deep_rnn_shared.detach()[None].repeat(self.num_states, 1, 1)
                deep_rnn_shared = self.linear_deep_to_rnn(deep_rnn_shared)
                initial_state = (deep_rnn_shared, deep_rnn_shared) if self.rnn_type == "LSTM" else deep_rnn_shared
            else:
                initial_state = None
            ht_link, arrival_pred = self.rnn_link(seq_link_dense, seq_link_sparse, num_links, initial_state)
            if self.use_seq_cross:
                ht_cross = self.rnn_cross(seq_cross_dense, num_crosses, initial_state)
                head_input = torch.cat([out_fm, out_deep, ht_link, ht_cross], dim=-1)
            else:
                head_input = torch.cat([out_fm, out_deep, ht_link], dim=-1)
        else:
            head_input = torch.cat([out_fm, out_deep], dim=-1)
            arrival_pred = None

        return self.head(head_input), arrival_pred
