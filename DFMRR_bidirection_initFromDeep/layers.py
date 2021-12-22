import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class SparseLinear(nn.Module):
    def __init__(self, field_dims, output_dim):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((1, output_dim)))
        self.offsets = torch.from_numpy(np.array((0, *np.cumsum(field_dims)[:-1]))).long()

    def forward(self, x):
        """
        Add embedding to sparse features and take sum <=> Linear combination of sparse features
        Args:
            x: (N, num_fields)
        Output:
            (N, output_dim)
        """
        x = x + self.offsets[None].to(x.device)
        return torch.sum(self.fc(x), dim=1) + self.bias


class SparseEmbedding(nn.Module):
    def __init__(self, field_dims, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), emb_dim)
        self.offsets = torch.from_numpy(np.array((0, *np.cumsum(field_dims)[:-1]))).long()
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        Add embedding to sparse features
        Args:
            x: (N, num_fields)
        Output:
            (N, num_fields, emb_dim)
        """
        x = x + self.offsets[None].to(x.device)
        return self.embedding(x)


class Wide(nn.Module):
    def __init__(self, dense_field_dims, sparse_field_dims, output_dim):
        '''
        dense_field_dims: the dimensionality of dense features
        sparse_field_dims: a list containing dimensionality of each field of sparse features
        sparse_cross_field_dims: a list containing dimensionality of each field of sparse cross features
        '''
        super().__init__()
        self.dense_linear = nn.Linear(dense_field_dims, output_dim)
        self.sparse_linear = SparseLinear(sparse_field_dims, output_dim)
        # self.sparse_cross_linear = SparseLinear(sparse_cross_field_dims, output_dim)

    def forward(self, dense, sparse):
        '''
        Args:
            dense: (N, dense_field_dims)
            sparse: (N, len(sparse_field_dims))
            sparse_cross: (N, len(sparse_cross_field_dims))
        Output:
            (N, output_dim)
        '''
        # sparse already contains cross-product terms
        # return F.relu(self.dense_linear(dense) + self.sparse_linear(sparse) + self.sparse_cross_linear(sparse_cross))
        return F.relu(self.dense_linear(dense) + self.sparse_linear(sparse))


def factorization_machine(x):
    """
    Args:
        x: (N, num_fields, emb_dim)
    Output:
        (N, 1)
    """
    square_of_sum = torch.sum(x, dim=1) ** 2
    sum_of_square = torch.sum(x ** 2, dim=1)
    ix = square_of_sum - sum_of_square
    return 0.5 * torch.sum(ix, dim=1, keepdim=True)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, batchnorm, dropout):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (N, emb_dim)
        """
        return self.mlp(x)


class RecurrentLink(nn.Module):
    def __init__(self, hidden_dim, num_layers, bidirectional, dropout, rnn_type, init_from_deep,
                 seq_dense_features, seq_sparse_features, num_embs, emb_dim, output_layer):
        '''
        Args:
            init_from_deep (bool): whether or not use embeddings from deep to initialize the RNN
            seq_dense_features: a list containing names of sequantial dense features
            seq_sparse_features: a list containing names of sequantial sparse features
            num_embs: a dictionary containing number of embeddings for sparse features
            emb_dim: embedding dimension, which is the same for each sparse feature
            output_layer (bool): if auxiliary loss, this is True
        '''
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.init_from_deep = init_from_deep
        self.rnn_type = rnn_type
        self.output_layer = output_layer
        self.sparse_emb = nn.ModuleList([nn.Embedding(num_embs[sp], emb_dim) for sp in seq_sparse_features])
        self.linear = nn.Linear(len(seq_dense_features) + emb_dim * len(seq_sparse_features), hidden_dim)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout, bidirectional=bidirectional)
        else:
            raise NotImplementedError

        if self.output_layer:
            D = 2 if self.bidirectional else 1
            self.linear_pred = nn.Linear(hidden_dim * D, 4)

    def forward(self, seq_dense, seq_sparse, num_links, initial_state=None):
        '''
        Args:
            seq_dense: padded sequence (N, T, d_dense)
            seq_sparse: padded sequence (N, T, d_sparse)
            num_links
            initial_state: (h, c) at time 0
        Output:
            h: (N, hidden_dim*D)
            output: (sum(num_links), 5)
        '''
        seq_sparse = torch.cat([emb(seq_sparse[..., i]) for i, emb in enumerate(self.sparse_emb)], dim=-1)
        seq = torch.cat([seq_dense, seq_sparse], dim=-1)
        seq = F.relu(self.linear(seq))
        seq = pack_padded_sequence(seq, num_links.cpu(), batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        if self.init_from_deep:
            output, h = self.rnn(seq, initial_state)
        else:
            output, h = self.rnn(seq)

        if self.rnn_type == "LSTM":
            h = h[0]

        h = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
        if self.output_layer:
            output = self.linear_pred(output.data)
        else:
            output = output.data
        return h, output


class RecurrentCross(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, dropout, rnn_type, init_from_deep):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.init_from_deep = init_from_deep
        self.rnn_type = rnn_type
        self.linear = nn.Linear(input_dim, hidden_dim)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=dropout, bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout, bidirectional=bidirectional)
        else:
            raise NotImplementedError

    def forward(self, cross_dense, num_crosses, initial_state=None):
        '''
        Args:
            cross_dense: padded sequence (N, T, d_dense)
            initial_state: (h, c) at time 0
        Output:
            h: (N, hidden_dim*D)
            output: (sum(num_links), 5)
        '''
        seq = F.relu(self.linear(cross_dense))
        seq = pack_padded_sequence(seq, num_crosses.cpu(), batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        if self.init_from_deep:
            _, h = self.rnn(seq, initial_state)
        else:
            _, h = self.rnn(seq)

        if self.rnn_type == "LSTM":
            h = h[0]

        h = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
        return h
