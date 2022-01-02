import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import WideDeepRecurrentRecurrent, DeepFMRecurrentRecurrent


# class NoamOpt:
#     """
#     learning rate warmup and decay
#     """
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0

#     def step(self):
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()

#     def rate(self, step=None):
#         if step is None:
#             step = self._step
#         return self.factor * \
#             (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.auxiliary_loss = configs.rnn['auxiliary_loss']
        torch.manual_seed(5)
        if configs.model == "WDRR":
            self.model = WideDeepRecurrentRecurrent(configs)
        elif configs.model == "DeepFM":
            self.model = DeepFMRecurrentRecurrent(configs)
        else:
            raise NotImplementedError

        if configs.use_multi_gpu_for_train:
            print(f"will use {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model).to(configs.device)
        else:
            print(f"will use single GPU for training")
            self.model = self.model.to(configs.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.3, patience=0, verbose=True, min_lr=configs.min_lr)

    def loss_func(self, y_pred, y_true):
        return torch.mean((y_pred - y_true)**2)

    def loss_mape(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true) / (y_true + 1e-7))

    def loss_aux(self, arrival_pred, arrival_true):
        arrival_pred = arrival_pred[arrival_true > 0]
        arrival_true = arrival_true[arrival_true > 0] - 1
        return F.cross_entropy(arrival_pred, arrival_true)

    def retrieve(self, x):
        dense = x['dense'].to(self.device)
        sparse = x['sparse'].to(self.device)
        sparse_cross = x['sparse_cross'].to(self.device) if x['sparse_cross'] is not None else None
        seq_link_dense = x['seq_link_dense'].to(self.device) if x['seq_link_dense'] is not None else None
        seq_link_sparse = x['seq_link_sparse'].to(self.device) if x['seq_link_sparse'] is not None else None
        num_links = x['num_links']
        seq_cross_dense = x['seq_cross_dense'].to(self.device) if x['seq_cross_dense'] is not None else None
        num_crosses = x['num_crosses']
        links_arrival_status = x['links_arrival_status'].data.squeeze().to(self.device) if x['links_arrival_status'] is not None else None
        ata = x['ata'].to(self.device)
        return dense, sparse, sparse_cross,\
               seq_link_dense, seq_link_sparse,\
               num_links, seq_cross_dense, num_crosses,\
               links_arrival_status, ata

    def train_one_batch(self, x):
        *data, arrival_true, ata = self.retrieve(x)
        eta, arrival_pred = self.model(*data)
        self.opt.zero_grad()
        loss = self.loss_mape(eta, ata)
        if self.auxiliary_loss:
            aux_loss = self.loss_aux(arrival_pred, arrival_true)
            (loss + 0.1 * aux_loss).backward()
            aux_loss = aux_loss.item()
        else:
            aux_loss = None
            loss.backward()
        if self.configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.clipping_threshold)
        self.opt.step()
        return loss.item(), aux_loss

    def evaluate(self, dataloader):
        self.model.eval()
        size = 0
        loss_sum = 0
        with torch.no_grad():
            for x in dataloader:
                *data, _, ata = self.retrieve(x)
                eta, _ = self.model(*data)
                size += eta.size(0)
                loss_sum += (self.loss_mape(eta, ata).item() * eta.size(0))
        return loss_sum / size

    def train(self, dataloader_train, dataloader_eval):
        # torch.manual_seed(0)
        # print('loading train dataloader')
        # dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        # print('loading eval dataloader')
        # dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i + 1))
            # train
            self.model.train()
            epoch_start_time = time.time()
            for j, x in enumerate(dataloader_train):
                loss_train, aux_loss = self.train_one_batch(x)
                if j % self.configs.display_interval == 0:
                    if aux_loss is not None:
                        print('batch training loss: {:.5f}, aux_loss: {:.5f}'.format(loss_train, aux_loss))
                    else:
                        print('batch training loss: {:.5f}'.format(loss_train))
            # evaluation
            epoch_training_time = self.get_elapsed_time(epoch_start_time)
            eval_start_time = time.time()
            loss_eval = self.evaluate(dataloader_eval)
            eval_time = self.get_elapsed_time(eval_start_time)
            print('epoch training time: {}\nepoch eval loss: {:.5f}, eval time: {}'.format(epoch_training_time, loss_eval, eval_time))
            self.lr_scheduler.step(loss_eval)
            if loss_eval >= best:
                count += 1
                print('eval loss is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval loss is improved from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model()
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, path):
        with open(path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self):
        torch.save({'net': self.model.state_dict(),
                    'optimizer': self.opt.state_dict()}, self.configs.chk_path)

    def get_elapsed_time(self, start_time):
        elapsed_time = time.time() - start_time
        return str(round(elapsed_time, 3)) + ' s' if elapsed_time < 60 else str(round(elapsed_time / 60, 3)) + "mins"
