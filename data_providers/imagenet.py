import shutil
import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
from data_providers.base_provider import *


class ImagenetDataProvider(DataProvider):
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=24, local_rank=0, world_size=1):

        self._save_path = save_path

        self.valid = None
        self.train = utils.get_data_iter(type='train', image_dir=self.train_path,
                                         batch_size=train_batch_size, num_threads=n_worker,
                                         device_id=local_rank,
                                         num_gpus=torch.cuda.device_count(), crop=self.image_size,
                                         val_size=self.resize_value, world_size=world_size, local_rank=local_rank)
        self.test = utils.get_data_iter(type='val', image_dir=self.valid_path,
                                        batch_size=test_batch_size, num_threads=n_worker, device_id=local_rank,
                                        num_gpus=torch.cuda.device_count(), crop=self.image_size,
                                        val_size=self.resize_value, world_size=world_size, local_rank=local_rank)
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/userhome/data/imagenet'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224
