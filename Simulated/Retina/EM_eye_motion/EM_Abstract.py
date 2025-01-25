import torch.nn as nn
from root_config import *
from abc import ABC, abstractmethod

class AbstractEyeMotion(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractEyeMotion, self).__init__()

    @abstractmethod
    def forward(self, LMS_full_field):
        # Input: LMS_full_field image (batch_size, C, H, W)
        # TODO: Output: batch_LMS_current_FoV, batch_true_dxy
        pass


