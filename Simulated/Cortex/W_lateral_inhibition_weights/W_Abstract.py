import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractLateralInhibitionWeights(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractLateralInhibitionWeights, self).__init__()

    @abstractmethod
    def deconvolve(self, ons):
        pass

    @abstractmethod
    def convolve(self, pa):
        pass
