import torch
import numpy as np
import normflows as nf
import torch.nn.functional as F
from Simulated.Cortex.P_cell_position.P_Abstract import AbstractCellPosition
from . import register_class


@register_class("Default")
class DefaultCellPosition(AbstractCellPosition):
    def __init__(self, params, device):
        super(DefaultCellPosition, self).__init__(params, device)

        self.device = device
        self.simulation_size = params['Experiment']['simulation_size']
        
        x_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        y_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        xx_ons_dim, yy_ons_dim = torch.meshgrid(x_ons_dim, y_ons_dim, indexing='xy')
        regular_grid = torch.stack([xx_ons_dim, yy_ons_dim], 0).unsqueeze(0).to(device=self.device, memory_format=torch.channels_last)
        self.regular_grid_uv = (regular_grid - ((self.simulation_size-1)/2)) / ((self.simulation_size-1)/2)

        # Initialize RealNVP model as an invertible coordinate-based transformation
        latent_dim = 16
        num_layers = 8
        flows = []
        for _ in range(num_layers):
            param_map = nf.nets.MLP([1, latent_dim, latent_dim, 2], init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
        base = nf.distributions.base.DiagGaussian(2)
        self.RealNVP = nf.NormalizingFlow(base, flows).to(self.device)

        batch_size = params['Dataset']['batch_size']
        timesteps_per_image = params['Experiment']['timesteps_per_image']

        self.mask = torch.ones((batch_size * (timesteps_per_image-1), 1, self.simulation_size, self.simulation_size)).to(device=self.device, memory_format=torch.channels_last)


    # Get cone locations in the retina space (XY)
    def get_XY_default_locations(self):
        return self.RealNVP.forward(self.regular_grid_uv.permute(0,2,3,1).reshape(-1, 2)).reshape(-1, self.simulation_size, self.simulation_size, 2).permute(0,3,1,2)


    # Get the corresponding UV coordinates for the given XY locations
    def get_UV_locations(self, xy_locations):
        (BS, _, H, W) = xy_locations.shape
        return self.RealNVP.inverse(xy_locations.permute(0,2,3,1).reshape(-1, 2)).reshape(BS, H, W, 2).permute(0,3,1,2)
    

    # Efficiently transform the warped internal percept image to the new warped image, based on the predicted dxy
    def efficient_warping(self, ip1, pred_dxy):
        xy = self.get_XY_default_locations() # current cone location estimate
        translated1 = xy + pred_dxy.reshape(-1, 2, 1, 1)

        grid1 = self.get_UV_locations(translated1).permute(0,2,3,1)
        ip2_pred = F.grid_sample(ip1, grid1, align_corners=True, mode='bilinear', padding_mode='zeros')
        mask2 = F.grid_sample(self.mask, grid1, align_corners=True, mode='nearest', padding_mode='zeros')
        
        return ip2_pred, mask2