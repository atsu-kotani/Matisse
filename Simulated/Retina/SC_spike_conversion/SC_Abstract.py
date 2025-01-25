import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractSpikeConversion(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractSpikeConversion, self).__init__()

    @abstractmethod
    def forward(self, bipolar_signals):
        pass
