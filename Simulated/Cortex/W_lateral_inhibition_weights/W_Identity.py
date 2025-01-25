import torch
import torch.nn as nn
from root_config import *
import torch.nn.functional as F
from Simulated.Cortex.W_lateral_inhibition_weights.W_Abstract import AbstractLateralInhibitionWeights
from . import register_class

@register_class("Identity")
class IdentityLateralInhibitionWeights(AbstractLateralInhibitionWeights):
    def __init__(self, params, device):
        super(IdentityLateralInhibitionWeights, self).__init__(params, device)


    def deconvolve(self, ons):
        return ons


    def convolve(self, pa):
        return pa


    def get_predicted_kernel(self, kernel_size):
        pred_kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        pred_kernel[center, center] = 1
        return pred_kernel