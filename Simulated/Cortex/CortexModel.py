import torch
import torch.nn as nn
import torch.nn.functional as F
from Experiment.helper import *
from Simulated.Cortex.W_lateral_inhibition_weights import create_W_lateral_inhibition_weights
from Simulated.Cortex.C_cone_spectral_type import create_C_cone_spectral_type
from Simulated.Cortex.D_demosaicing import create_D_demosaicing
from Simulated.Cortex.P_cell_position import create_P_cell_position
from Simulated.Cortex.M_global_movement import create_M_global_movement
from Simulated.NeuralScope.NS_cone_mosaic import NS_cone_mosaic
from Simulated.NeuralScope.NS_internal_percept import NS_internal_percept


class CortexModel(nn.Module):
    def __init__(self, params, device):
        super(CortexModel, self).__init__()

        self.C_cone_spectral_type         = create_C_cone_spectral_type(params, device)
        self.D_demosaicing                = create_D_demosaicing(params, device)
        self.M_global_movement            = create_M_global_movement(params, device)
        self.P_cell_position              = create_P_cell_position(params, device)
        self.W_lateral_inhibition_weights = create_W_lateral_inhibition_weights(params, device)

        self.ns_ip = NS_internal_percept(latent_dim=params['CorticalModel']['latent_dim'], output_dim=3).to(device)
        self.ns_cm = NS_cone_mosaic(latent_dim=params['CorticalModel']['latent_dim'], output_dim=4).to(device)

        self.device = device

        # Disabling neural scope for internal percept when simulating the tetrachromatic model
        if params['Experiment']['simulating_tetra']:
            self.learn_ip_ns = False
        else:
            self.learn_ip_ns = True


    def decode(self, ons):
        pa = self.W_lateral_inhibition_weights.deconvolve(ons)
        C_injected_pa = self.C_cone_spectral_type.C_injection(pa)
        ip = self.D_demosaicing.demosaic(C_injected_pa)
        return ip
    

    def encode(self, ip):
        pa = self.C_cone_spectral_type.C_sampling(ip)
        ons = self.W_lateral_inhibition_weights.convolve(pa)
        return ons


    def main_train(self, ons1, ons2, linsRGB1, true_dxy, cone_mosaic, kernel_size):

        warped_ip1 = self.decode(ons1)

        P = kernel_size // 2
        pred_dxy = self.M_global_movement.forward(ons1, ons2, self.P_cell_position, true_dxy)
        pred_warped_ip2, mask2 = self.P_cell_position.efficient_warping(warped_ip1, pred_dxy)

        ons2_pred = self.encode(pred_warped_ip2)

        main_loss = torch.sum(torch.sum((((ons2_pred[:,:,P:-P,P:-P] - ons2[:,:,P:-P,P:-P])**2) * mask2[:,:,P:-P,P:-P]),0) / (torch.sum(mask2[:,:,P:-P,P:-P],0)+1))

        # neural scope
        C = self.C_cone_spectral_type.get_cone_indetity_function().detach().clone()
        pred_cone_mosaic = self.ns_cm(C)
        ns_cm_loss = torch.sum((pred_cone_mosaic - cone_mosaic)[:,:,P:-P,P:-P] ** 2)

        if self.learn_ip_ns:
            warped_ip1 = warped_ip1.detach().clone()
            pred_warped_linsRGB1 = self.ns_ip(warped_ip1)
            ns_ip_loss = torch.sum((pred_warped_linsRGB1[:,:3] - linsRGB1[:,:3])[:,:,P:-P,P:-P] ** 2)
        else:
            ns_ip_loss = 0

        return main_loss, ns_cm_loss, ns_ip_loss, pred_cone_mosaic, (ons2_pred * mask2), pred_dxy
    

    def get_unwarped_percept(self, warped_ip_sRGB):
        xy_full = self.P_cell_position.get_XY_default_locations()
        required_image_resolution = compute_required_image_resolution(xy_full.detach().clone())

        grid = self.M_global_movement.generate_grid_fixed(xy_full[0,:,0,0], xy_full[0,:,-1,0], xy_full[0,:,0,-1], xy_full[0,:,-1,-1], required_image_resolution)
        uvs = self.P_cell_position.get_UV_locations(grid.permute(2,0,1).unsqueeze(0))

        if self.device == 'mps:0':
            # F.grid_sample function acts strangely on MPS, so we move to CPU
            warped_ip_sRGB = warped_ip_sRGB.to('cpu')
            uvs = uvs.to('cpu')

        uvs = uvs.repeat(len(warped_ip_sRGB),1,1,1)
        # Image.fromarray(np.uint8(warped_ip_sRGB[4].permute(1,2,0).cpu().detach().numpy() * 255)).save('a.png')

        internal_percept_sRGB = F.grid_sample(warped_ip_sRGB, uvs.permute(0,2,3,1), align_corners=True, mode='bilinear', padding_mode='zeros')

        internal_percept_sRGB = internal_percept_sRGB.permute(0,2,3,1).detach().cpu().numpy()
        internal_percept_sRGB = np.clip(internal_percept_sRGB, 0, 1)
        invalid_regions = (((uvs <= -1) | (uvs >= 1)).sum(1) > 0).cpu().detach().numpy()
        invalid_regions = invalid_regions[0].astype(int)

        return internal_percept_sRGB, invalid_regions