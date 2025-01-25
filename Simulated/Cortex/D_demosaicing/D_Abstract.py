import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractDemosaicing(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractDemosaicing, self).__init__()

    @abstractmethod
    def demosaic(self, C_encoded_pa):
        pass
