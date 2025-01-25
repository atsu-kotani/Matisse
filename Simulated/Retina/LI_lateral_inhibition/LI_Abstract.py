import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractLateralInhibition(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractLateralInhibition, self).__init__()

    @abstractmethod
    def forward(self, photoreceptor_activation):
        # Requirement: output should be a tensor of (BS, 1, params.ons_dim, params.ons_dim)
        # Input: photoreceptor activation (BS, 1, params.ons_dim, params.ons_dim)
        # Output: lateral inhibition applied photoreceptor activation (i.e. bipolar signals) (BS, 1, params.ons_dim, params.ons_dim)
        pass
    
    @abstractmethod
    def get_kernel_size(self):
        pass
