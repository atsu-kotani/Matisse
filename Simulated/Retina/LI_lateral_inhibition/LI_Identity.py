import torch
from Simulated.Retina.LI_lateral_inhibition.LI_Abstract import AbstractLateralInhibition
from . import register_class

@register_class("Identity")
class IdentityLateralInhibition(AbstractLateralInhibition):
    def __init__(self, params, device):
        super(IdentityLateralInhibition, self).__init__(params, device)

    # Identity function
    def forward(self, pa):
        return pa


    def get_kernel_size(self):
        return 3


    def get_LI_kernel(self):
        kernel = torch.zeros(3,3)
        kernel[1,1] = 1
        return kernel