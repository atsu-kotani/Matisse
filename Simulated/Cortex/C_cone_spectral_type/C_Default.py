import torch
import torch.nn as nn
from Simulated.Cortex.C_cone_spectral_type.C_Abstract import AbstractConeSpectralType
from . import register_class

@register_class("Default")
class DefaultConeSpectralType(AbstractConeSpectralType):
    def __init__(self, params, device):
        super(DefaultConeSpectralType, self).__init__(params, device)

        self.device = device
        latent_dim = params['CorticalModel']['latent_dim']
        simulation_size = params['Experiment']['simulation_size']

        cone_indetity_function = torch.zeros([1, latent_dim, simulation_size, simulation_size]).to(self.device)
        cone_indetity_function[0,0,:,:] = 1
        self.raw_cone_indetity_function = nn.Parameter(cone_indetity_function)


    def get_cone_indetity_function(self):
        C = self.raw_cone_indetity_function
        C = C / (torch.sqrt(torch.sum(C * C, 1, keepdim=True)) + 1e-5)
        return C


    def C_injection(self, pa):
        C = self.get_cone_indetity_function()
        C_injected_pa = pa * C
        return C_injected_pa


    def C_sampling(self, C_injected_pa):
        C = self.get_cone_indetity_function()
        pa = torch.sum(C_injected_pa * C, 1, keepdim=True)
        return pa