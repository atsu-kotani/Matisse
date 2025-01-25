import torch
import numpy as np
from root_config import *
from Simulated.Retina.SS_spectral_sampling.SS_Abstract import AbstractSpectralSampling
from Simulated.Retina.SS_spectral_sampling.helper.generate_cone_fundamentals import generate_cone_fundamentals_from_peak_frequencies
from Simulated.Retina.SS_spectral_sampling.helper.generate_cone_mosaic import generate_default_cone_mosaic
from Simulated.Retina.SS_spectral_sampling import register_class


@register_class("Custom_Cone_Fundamentals")
class CustomSpectralSampling(AbstractSpectralSampling):
    def __init__(self, params, device):
        super(CustomSpectralSampling, self).__init__(params, device)
        
        cone_mosaic = generate_default_cone_mosaic(params)
        cone_mosaic = torch.FloatTensor(cone_mosaic).to(device)
        self.cone_mosaic = cone_mosaic.permute(2,0,1).unsqueeze(0).unsqueeze(0) # (1, 1, 4, H, W)

        # Cone cell activation noise
        self.cone_cell_noise_std = 0.01

        # Custom cone fundamentals
        cone_fundamentals_params = params['RetinaModel']['retina_spectral_sampling']['cone_fundamentals']
        cone_fundamental_peaks = list(cone_fundamentals_params.values())

        self.cone_types = list(cone_fundamentals_params.keys())
        cone_fundamentals = generate_cone_fundamentals_from_peak_frequencies(cone_fundamental_peaks)
        self.cone_fundamentals = torch.FloatTensor(cone_fundamentals).to(device)
        

    def forward(self, lms):
        
        photoreceptor_activation = torch.sum(lms * self.cone_mosaic, 2)

        ratio = np.random.randint(0, 11) * 0.1
        noisy_pa = torch.normal(photoreceptor_activation, torch.sqrt(photoreceptor_activation) * self.cone_cell_noise_std * ratio + 1e-10)
        return noisy_pa


    def get_cone_fundamentals(self):
        return self.cone_fundamentals


    def get_cone_mosaic(self):
        return self.cone_mosaic[0]
