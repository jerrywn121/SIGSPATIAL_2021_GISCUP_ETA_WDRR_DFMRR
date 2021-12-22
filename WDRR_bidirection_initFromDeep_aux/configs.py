import torch
from pathlib import Path


class Configs:
    def update(self, model=None, use_rnn=None, rnn_type=None, bidirectional=None, init_from_deep=None):
        if model is not None:
            assert model == "WDRR" or model == "DeepFM"
            self.model = model
        if use_rnn is not None:
            self.rnn['use_seq_link'] = use_rnn
            self.rnn['use_seq_cross'] = use_rnn
            if use_rnn:
                if bidirectional is not None:
                    self.rnn['bidirectional'] = bidirectional
                if init_from_deep is not None:
                    self.rnn['init_from_deep'] = init_from_deep
        configs.rnn['output_dim'] = configs.rnn['hidden_dim'] * (2 if configs.rnn['bidirectional'] else 1)
        configs.rnn['auxiliary_loss'] = configs.rnn['use_seq_link'] and configs.rnn['auxiliary_loss']
        configs.rnn['init_from_deep'] = configs.rnn['use_seq_link'] and configs.rnn['init_from_deep']
        configs.rnn['use_seq_cross'] = configs.rnn['use_seq_link'] and configs.rnn['use_seq_cross']


configs = Configs()

# ----- trainer -----
configs.n_cpu = 32
configs.device = torch.device('cuda:0')  # torch.device("cpu")
configs.batch_size_test = 4096*2
configs.batch_size = 4096
configs.use_multi_gpu_for_train = True
configs.lr = 0.001
configs.min_lr = 0.0001
configs.weight_decay = 0
configs.display_interval = 90
configs.num_epochs = 50
configs.early_stopping = True
configs.patience = 4
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# ----- data -----
configs.data_dir = Path('/user-data/data_16_31')
configs.chk_path = "checkpoint.chk"
configs.all_features = {'dense': ['distance', 'simple_eta', 'links_time_total', 'crosses_time_total'],
                        'sparse': ['time_slice_id_5m', 'time_slice_id_30m', 'day_of_week',
                                   'is_holiday', 'weather'],
                        'seq_link': {'dense': ['links_time', 'links_ratio'],
                                     'sparse': ['links_current_status', 'is_cross',
                                                'num_next_links', 'num_prev_links']},
                        'seq_cross': {'dense': ['crosses_time'],
                                      'sparse': []}}

configs.features_to_use = {'dense': ['distance', 'simple_eta', 'links_time_total', 'crosses_time_total'],
                           'sparse': ['time_slice_id_30m', 'day_of_week', 'weather'],
                           'seq_link': {'dense': ['links_time', 'links_ratio'],
                                        'sparse': ['links_current_status', 'is_cross',
                                                   'num_next_links', 'num_prev_links']},
                           'seq_cross': {'dense': ['crosses_time'],
                                         'sparse': []}}

# all emb_dim should be the same (for convenience)
configs.emb_dim = 20
# number of embeddings for sparse features
configs.sparse_num_emb = {'time_slice_id_5m': 288, 'time_slice_id_30m': 48, 'day_of_week': 7,
                          'is_holiday': 2, 'weather': 5, 'links_current_status': 5,
                          'is_cross': 3, 'num_next_links': 8, 'num_prev_links': 8}

# number of embeddings for cross product of non-link-and-cross sparse features
# sp = configs.features_to_use['sparse']
# configs.sparse_cross_num_emb = {}
# configs.sparse_cross_transform = {}
# for i in range(len(sp)):
#     for j in range(i+1, len(sp)):
#         num_embi = configs.sparse_num_emb[sp[i]]
#         num_embj = configs.sparse_num_emb[sp[j]]
#         emb_dict = [f"{a}+{b}" for a in range(num_embi) for b in range(num_embj)]
#         configs.sparse_cross_transform[f"{sp[i]}+{sp[j]}"] = dict(zip(emb_dict, list(range(len(emb_dict)))))
#         configs.sparse_cross_num_emb[f"{sp[i]}+{sp[j]}"] = num_embi * num_embj


configs.dense_field_dims = len(list(configs.features_to_use['dense']))
configs.sparse_field_dims = [configs.sparse_num_emb[x] for x in configs.features_to_use['sparse']]
# configs.sparse_cross_field_dims = list(configs.sparse_cross_num_emb.values())

configs.train_period = ['20200816', '20200827']
configs.eval_period = ['20200828', '20200829']
configs.test_period = ['20200830', '20200831']


# ----- model -----
configs.model = "WDRR"  # "WDRR" "DeepFM"
assert configs.model == "WDRR" or configs.model == "DeepFM"

configs.wide = {'output_dim': 256}
configs.deep = {'hidden_dims': [256, 256], 'output_dim': 256,
                'batchnorm': False, 'dropout': 0.1}
configs.rnn = {'use_seq_link': True, 'use_seq_cross': True,
               'hidden_dim': 256, 'num_layers': 2, 'type': "LSTM",
               'bidirectional': True, 'init_from_deep': True,
               'auxiliary_loss': True, 'emb_dim': configs.emb_dim}
configs.rnn['output_dim'] = configs.rnn['hidden_dim'] * (2 if configs.rnn['bidirectional'] else 1)
configs.rnn['auxiliary_loss'] = configs.rnn['use_seq_link'] and configs.rnn['auxiliary_loss']
configs.rnn['init_from_deep'] = configs.rnn['use_seq_link'] and configs.rnn['init_from_deep']
configs.rnn['use_seq_cross'] = configs.rnn['use_seq_link'] and configs.rnn['use_seq_cross']

configs.head = {'head_hidden_dim': 32}






