import pandas as pd
# from pandarallel import pandarallel
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import datetime
from pathlib import Path


def date_range(start_date, end_date):
    assert len(start_date) == len(end_date) == 8
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    dates = []
    curr = start_date
    while curr <= end_date:
        dates.append(curr.strftime("%Y%m%d"))
        curr += datetime.timedelta(1)
    return dates


def load_files(data_dir, start_date, end_date, cols):
    dates = date_range(start_date, end_date)
    print(f"loading files from {dates}")
    files = [Path(data_dir) / f"{date}.parquet" for date in dates]
    df = pd.concat([pd.read_parquet(file, columns=cols) for file in files], axis=0, ignore_index=True)
    return df


def cross_product_transform(row, x, y, transform):
    '''
    used for cross product transformation of sparse features
    Args:
        row: a row in pandas dataframe
        transform: a dictionary
    '''
    return transform[f"{x}+{y}"][f"{row[x]}+{row[y]}"]


def seq_to_tensor(x):
    return torch.tensor([i for i in x]).T


class OrderDataset(Dataset):
    def __init__(self, data_dir, start_date, end_date, configs):
        self.configs = configs
        features = configs.features_to_use
        cols = features['dense'] + features['sparse'] + ['ata']
        if configs.rnn['use_seq_link']:
            cols += features['seq_link']['dense']
            cols += features['seq_link']['sparse']
        if configs.rnn['use_seq_cross']:
            cols += features['seq_cross']['dense']
            cols += features['seq_cross']['sparse']  # this should generally be []
        if configs.rnn['auxiliary_loss']:
            cols += ['links_arrival_status']

        df = load_files(data_dir, start_date, end_date, cols)

        self.dense = df[features['dense']].values
        self.sparse = df[features['sparse']].values
        # if configs.model == "WDRR":
        #     sp = features['sparse']
        #     self.sparse_cross = []
        #     pandarallel.initialize()
        #     for i in range(len(sp)):
        #         for j in range(i+1, len(sp)):
        #             self.sparse_cross.append(df.parallel_apply(cross_product_transform, x=sp[i], y=sp[j],
        #                                                        transform=self.configs.sparse_cross_transform, axis=1).values)
        #     self.sparse_cross = np.stack(self.sparse_cross, axis=1)

        self.use_seq_link = configs.rnn['use_seq_link']
        self.use_seq_cross = configs.rnn['use_seq_cross']
        self.auxiliary_loss = configs.rnn['auxiliary_loss']
        if self.use_seq_link:
            seq_link_features = features['seq_link']
            self.seq_link_dense = df[seq_link_features['dense']].values
            self.seq_link_sparse = df[seq_link_features['sparse']].values
            self.num_links = np.array([len(x[0]) for x in self.seq_link_dense])
        if self.use_seq_cross:
            seq_cross_features = features['seq_cross']
            self.seq_cross_dense = df[seq_cross_features['dense']].values
            self.num_crosses = np.array([len(x[0]) for x in self.seq_cross_dense])
        if self.auxiliary_loss:
            self.links_arrival_status = df[['links_arrival_status']].values
        self.ata = df.ata.values[:, None]

    def info(self):
        return {'dense': self.dense.shape,
                'sparse': self.sparse.shape,
                'sparse_cross': "removed, will never be used",
                'seq_link_dense': len(self.seq_link_dense) if self.use_seq_link else "did not use",
                'seq_link_sparse': len(self.seq_link_sparse) if self.use_seq_link else "did not use",
                'seq_cross_dense': len(self.seq_cross_dense) if self.use_seq_cross else "did not use",
                'links_arrival_status': len(self.links_arrival_status) if self.auxiliary_loss else "did not use",
                'ata': self.ata.shape}

    def __len__(self):
        return len(self.dense)

    def __getitem__(self, index):
        return [self.dense[index],
                self.sparse[index],
                None,  # sparse_cross
                seq_to_tensor(self.seq_link_dense[index]) if self.use_seq_link else None,
                seq_to_tensor(self.seq_link_sparse[index]) if self.use_seq_link else None,
                self.num_links[index] if self.use_seq_link else None,
                seq_to_tensor(self.seq_cross_dense[index]) if self.use_seq_cross else None,
                self.num_crosses[index] if self.use_seq_cross else None,
                seq_to_tensor(self.links_arrival_status[index]) if self.auxiliary_loss else None,
                self.ata[index]]


def collate_fn(batch):
    dense, sparse, _, seq_link_dense,\
    seq_link_sparse, num_links, seq_cross_dense,\
    num_crosses, links_arrival_status, ata = [list(x) for x in zip(*batch)]
    dense = torch.from_numpy(np.stack(dense, axis=0)).float()
    sparse = torch.from_numpy(np.stack(sparse, axis=0)).long()
    # sparse_cross = torch.from_numpy(np.stack(sparse_cross, axis=0)).long() if sparse_cross[0] is not None else None
    seq_link_dense = pad_sequence(seq_link_dense, batch_first=True).float() if seq_link_dense[0] is not None else None
    seq_link_sparse = pad_sequence(seq_link_sparse, batch_first=True).long() if seq_link_sparse[0] is not None else None
    num_links = torch.tensor(num_links) if num_links[0] is not None else None
    seq_cross_dense = pad_sequence(seq_cross_dense, batch_first=True).float() if seq_cross_dense[0] is not None else None
    num_crosses = torch.tensor(num_crosses) if num_crosses[0] is not None else None
    links_arrival_status = pack_sequence(links_arrival_status, enforce_sorted=False) if links_arrival_status[0] is not None else None
    ata = torch.from_numpy(np.stack(ata, axis=0)).float()
    return {'dense': dense, 'sparse': sparse, 'sparse_cross': None,
            'seq_link_dense': seq_link_dense, 'seq_link_sparse': seq_link_sparse,
            'num_links': num_links, 'seq_cross_dense': seq_cross_dense, 'num_crosses': num_crosses,
            'links_arrival_status': links_arrival_status, 'ata': ata}
