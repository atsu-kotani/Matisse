import torch.utils.data as data
from abc import ABC, abstractmethod


class Dataset(data.Dataset, ABC):
    def __init__(self, params, retina):
        super(Dataset, self).__init__()

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass