import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractGlobalMovement(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractGlobalMovement, self).__init__()

    @abstractmethod
    def forward(self, ons1, ons2, P_cell_position):

        # Output: pred_dxy (BS, 2, 1, 1) in range [-1, 1]
        pass
