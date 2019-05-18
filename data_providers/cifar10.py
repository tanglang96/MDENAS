import shutil
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
from data_providers.base_provider import *
import torchvision.datasets as dset


class CifarDataProvider(DataProvider):
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=24, local_rank=0, world_size=1):

        self._save_path = save_path
        dset_cls = dset.CIFAR10
        self.valid = None
        trn_transform, val_transform = utils.preproc.data_transforms('cifar10', 16)
        train_data = dset_cls(root=self.save_path, train=True, download=True, transform=trn_transform)
        valid_data = dset_cls(root=self.save_path, train=False, download=True, transform=val_transform)
        self.train = torch.utils.data.DataLoader(train_data,
                                                 batch_size=train_batch_size,
                                                 shuffle=True,
                                                 num_workers=n_worker,
                                                 pin_memory=True)
        self.test = torch.utils.data.DataLoader(valid_data,
                                                batch_size=test_batch_size,
                                                shuffle=False,
                                                num_workers=n_worker,
                                                pin_memory=True)
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, 32, 32  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/userhome/data'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download cifar10')
