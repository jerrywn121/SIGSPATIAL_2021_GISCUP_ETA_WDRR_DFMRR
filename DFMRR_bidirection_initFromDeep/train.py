import torch
from trainer import Trainer
from configs import configs
from torch.utils.data import DataLoader
from data_loader import OrderDataset, collate_fn
import os


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print({u: v for u, v in configs.__dict__.items() if u != 'sparse_cross_transform'})
    data_dir = configs.data_dir
    print(f'reading data from {data_dir}')
    print('processing training set')
    dataset_train = OrderDataset(data_dir, *configs.train_period, configs)
    print(dataset_train.info())

    print('processing eval set')
    dataset_eval = OrderDataset(data_dir, *configs.eval_period, configs)
    print(dataset_eval.info())

    trainer = Trainer(configs)
    trainer.save_configs('config_train.pkl')
    torch.manual_seed(0)
    print('loading train dataloader')
    dataloader_train = DataLoader(dataset_train, batch_size=configs.batch_size,
                                  shuffle=True, collate_fn=collate_fn, num_workers=configs.n_cpu)
    print('loading eval dataloader')
    dataloader_eval = DataLoader(dataset_eval, batch_size=configs.batch_size_test,
                                 shuffle=False, collate_fn=collate_fn, num_workers=configs.n_cpu)

    trainer.train(dataloader_train, dataloader_eval)
    print("======training done======\n")

    del dataset_train, dataset_eval, dataloader_train, dataloader_eval

    print('processing test set')
    dataset_test = OrderDataset(data_dir, *configs.test_period, configs)
    print(dataset_test.info())
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test,
                                 shuffle=False, collate_fn=collate_fn, num_workers=configs.n_cpu)
    trainer.model.load_state_dict(torch.load(configs.chk_path)['net'])
    loss_test = trainer.evaluate(dataloader_test)
    print(f"loss test: {loss_test}")
    os.system("/root/shutdown")


