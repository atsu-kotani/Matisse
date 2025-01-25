import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractSpectralSampling(nn.Module, ABC):
    def __init__(self, params, device):
        super(AbstractSpectralSampling, self).__init__()

        
    @abstractmethod
    def forward(self, lms):
        # Requirement: output should be a tensor of (BS, 1, params.ons_dim, params.ons_dim)
        # Input: LMS(+optionally Q) signals (BS, C, params.ons_dim, params.ons_dim)
        # Output: photoreceptor activation (BS, 1, params.ons_dim, params.ons_dim)
        pass

    @abstractmethod
    def get_cone_fundamentals(self):
        # Return the cone fundamentals of size (301, C) where C is the number of cones
        # Why 301? --> It should sample the spectrum from 400nm to 700nm with a step of 1nm.
        pass

    @abstractmethod
    def get_cone_mosaic(self):
        # Return the cone mosaic of size (C, params.ons_dim, params.ons_dim) where C is the number of cones
        pass
