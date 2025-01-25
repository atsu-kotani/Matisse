import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractCellPosition(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractCellPosition, self).__init__()

    @abstractmethod
    def get_XY_default_locations(self):
        pass

    @abstractmethod
    def get_UV_locations(self, xy_locations):
        pass

    @abstractmethod
    def efficient_warping(self, ip1, pred_dxy):
        pass