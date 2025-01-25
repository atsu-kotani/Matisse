import torch
import torch.nn as nn
from root_config import *
import torch.nn.functional as F
from Simulated.Cortex.W_lateral_inhibition_weights.W_Abstract import AbstractLateralInhibitionWeights
from . import register_class

@register_class("Default")
class DefaultLateralInhibitionWeights(AbstractLateralInhibitionWeights):
    def __init__(self, params, device):
        super(DefaultLateralInhibitionWeights, self).__init__(params, device)

        self.device = device

        ONS_DIM = params['Experiment']['simulation_size']
        self.FFT_DIM = ONS_DIM * 2 - 1

        # Left and right padding for the kernel
        self.L_PAD = (self.FFT_DIM - ONS_DIM) // 2
        self.R_PAD = (self.FFT_DIM - ONS_DIM) - self.L_PAD

        # Kernel
        kernel = torch.zeros([self.FFT_DIM, self.FFT_DIM]).to(self.device)
        kernel[self.FFT_DIM//2, self.FFT_DIM//2] = 1
        kernel_fft = torch.fft.fft2(kernel)
        kernel_fft_shift = torch.fft.fftshift(kernel_fft)
        LIF = torch.stack([kernel_fft_shift.real, kernel_fft_shift.imag], -1)
        self.LIF = nn.Parameter(LIF)


    def deconvolve(self, ons):
        LIF = torch.view_as_complex(self.LIF)
        ons = F.pad(ons, (self.L_PAD, self.R_PAD, self.L_PAD, self.R_PAD))
        ons = torch.fft.fftshift(torch.fft.fft2(ons), (2,3))

        # division in the frequency domain
        pa = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ons / LIF, (2,3))), (2,3))
        pa = pa[:,:,self.L_PAD:-self.R_PAD,self.L_PAD:-self.R_PAD].real
        return pa


    def convolve(self, pa):
        LIF = torch.view_as_complex(self.LIF)
        pa = F.pad(pa, (self.L_PAD, self.R_PAD, self.L_PAD, self.R_PAD))
        pa = torch.fft.fftshift(torch.fft.fft2(pa), (2,3))

        # multiplication in the frequency domain
        ons = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(pa * LIF, (2,3))), (2,3))
        ons = ons[:,:,self.L_PAD:-self.R_PAD,self.L_PAD:-self.R_PAD].real
        return ons


    def get_predicted_kernel(self, kernel_size):
        LIF         = torch.view_as_complex(self.LIF)
        padding     = (self.FFT_DIM - kernel_size) // 2
        pred_kernel = (torch.fft.ifft2(torch.fft.ifftshift(LIF))).real[padding:-padding,padding:-padding]
        return pred_kernel