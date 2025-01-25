import torch
import numpy as np
from root_config import *
import torch.nn.functional as F
from Simulated.Retina.LI_lateral_inhibition.LI_Abstract import AbstractLateralInhibition
from . import register_class

@register_class("Default")
class DefaultLateralInhibition(AbstractLateralInhibition):
    def __init__(self, params, device):
        super(DefaultLateralInhibition, self).__init__(params, device)

        self.ONS_DIM = params['Experiment']['simulation_size']
        self.FFT_DIM = self.ONS_DIM * 2 - 1
        
        self.LI_noise_std = 0.01

        self.LI_center_r = 0.15
        self.LI_surround_r = 0.90

        self.kernel_size = int(np.ceil(self.LI_surround_r * 6))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        center = gaussian_kernel(kernel_size=self.kernel_size, sig=self.LI_center_r)
        surround = gaussian_kernel(kernel_size=self.kernel_size, sig=self.LI_surround_r)
        
        self.LI_center_surround_ratio = 0.91
        kernel = center - self.LI_center_surround_ratio * surround
        
        self.LI_kernel = torch.FloatTensor(kernel).reshape(self.kernel_size,self.kernel_size).to(device)

        # Converting the kernel to the frequency domain
        self.PAD = self.kernel_size // 2
        self.L_PAD = (self.FFT_DIM - self.ONS_DIM) // 2
        self.R_PAD = (self.FFT_DIM - self.ONS_DIM) - self.L_PAD
        self.P = (self.FFT_DIM - self.kernel_size) // 2
        padded_LI_kernel = F.pad(self.LI_kernel, (self.P,self.P,self.P,self.P), value=0)
        LI_kernel_in_freq_domain = torch.fft.fft2(padded_LI_kernel)
        self.LI_kernel_in_freq_domain = torch.fft.fftshift(LI_kernel_in_freq_domain)
        

    def forward(self, pa):
        # Padding the optic nerve signal
        # pa = F.pad(pa, (self.L_PAD, self.R_PAD, self.L_PAD, self.R_PAD), mode='reflect')
        pa = F.pad(pa, (self.L_PAD, self.R_PAD, self.L_PAD, self.R_PAD))

        # Converting the optic nerve signal to the frequency domain
        pa = torch.fft.fftshift(torch.fft.fft2(pa), (2,3))

        # Applying the lateral inhibition kernel
        ons = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(pa * self.LI_kernel_in_freq_domain, (2,3))), (2,3))
        ons = ons[:,:,self.L_PAD:-self.R_PAD,self.L_PAD:-self.R_PAD].real
        
        # Adding noise to the optic nerve signal
        ratio = np.random.randint(0, 11) * 0.1
        noisy_ons = ons + torch.randn_like(ons) * ons * self.LI_noise_std * ratio

        return noisy_ons
    

    def get_kernel_size(self):
        return self.kernel_size


    def get_LI_kernel(self):
        return self.LI_kernel


# 2D Gaussian kernel
def gaussian_kernel(kernel_size=5, sig=1.):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)