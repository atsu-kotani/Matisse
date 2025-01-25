import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractConeSpectralType(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractConeSpectralType, self).__init__()

    @abstractmethod
    def get_cone_indetity_function(self):
        pass

    @abstractmethod
    def C_injection(self, pa):
        pass

    @abstractmethod
    def C_sampling(self, C_injected_pa):
        pass