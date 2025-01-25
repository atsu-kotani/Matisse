import torch
import torch.nn as nn
from Simulated.Retina.EM_eye_motion import create_eye_motion_module
from Simulated.Retina.FV_spatial_sampling import create_spatial_sampling_module
from Simulated.Retina.SS_spectral_sampling import create_spectral_sampling_module
from Simulated.Retina.LI_lateral_inhibition import create_lateral_inhibition_module
from Simulated.Retina.SC_spike_conversion import create_spike_conversion_module
from Simulated.Retina.helper.ColorSpaceTransform import ColorSpaceTransform


class RetinaModel(nn.Module):
    def __init__(self, params, device='cpu'):
        super(RetinaModel, self).__init__()

        self.EyeMotion          = create_eye_motion_module(params, device)
        self.SpatialSampling    = create_spatial_sampling_module(params, device)
        self.SpectralSampling   = create_spectral_sampling_module(params, device)
        self.LateralInhibition  = create_lateral_inhibition_module(params, device)
        self.SpikeConversion    = create_spike_conversion_module( params, device)        

        self.required_image_resolution = self.SpatialSampling.required_image_resolution
        self.CST = ColorSpaceTransform(self.SpectralSampling.get_cone_fundamentals(), device)

        self.device = device

    def forward(self, batch_LMS_full_field, intermediate_outputs=False):
        # Eye motion module samples the input image to the current field of view by applying the eye motion
        batch_LMS_current_FoV, batch_true_dxy = self.EyeMotion.forward(batch_LMS_full_field, self.required_image_resolution)
        # Normalize the true eye motion to the range of [-1, 1]
        batch_true_dxy = batch_true_dxy.float() / (self.required_image_resolution / 2)

        # Spatial sampling module warps the sampled image to the current field of view (foveation effect)
        batch_warped_LMS_current_FoV = self.SpatialSampling.forward(batch_LMS_current_FoV)

        # Spectral sampling module converts the warped LMS image to the photoreceptor activation values
        batch_pa = self.SpectralSampling.forward(batch_warped_LMS_current_FoV)

        # Lateral inhibition module applies the lateral inhibition effect to the photoreceptor activation values
        batch_bipolar_signals = self.LateralInhibition.forward(batch_pa)

        # Spike conversion module converts the bipolar signals to the spike signals (i.e. optic nerve signals)
        batch_ons = self.SpikeConversion.forward(batch_bipolar_signals)

        if not intermediate_outputs:
            return batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV
        else:
            return batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV, batch_pa, batch_bipolar_signals, batch_LMS_current_FoV
