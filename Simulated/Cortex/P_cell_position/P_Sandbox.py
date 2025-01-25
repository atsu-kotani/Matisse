import torch
import numpy as np
import normflows as nf
from root_config import DEVICE
import torch.nn.functional as F
from Simulated.Cortex.P_cell_position.P_Abstract import AbstractCellPosition
from . import register_class

@register_class("Sandbox")
class SandboxCellPosition(AbstractCellPosition):
    def __init__(self, params, device):
        super(SandboxCellPosition, self).__init__(params, device)

        self.device = device
        self.simulation_size = params['Experiment']['simulation_size']
        
        x_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        y_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        xx_ons_dim, yy_ons_dim = torch.meshgrid(x_ons_dim, y_ons_dim, indexing='xy')
        regular_grid = torch.stack([xx_ons_dim, yy_ons_dim], 0).unsqueeze(0).to(device=self.device, memory_format=torch.channels_last)
        self.regular_grid_uv = (regular_grid - ((self.simulation_size-1)/2)) / ((self.simulation_size-1)/2)

        self.MSS = params['RetinaModel']['max_shift_size']
        self.mask = torch.ones((1, self.simulation_size, self.simulation_size)).to(device=self.device)
        self.padded_mask = F.pad(self.mask, (self.MSS, self.MSS, self.MSS, self.MSS), "constant", 0)


    # Get cone locations in the retina space (XY)
    def get_XY_default_locations(self):
        return self.regular_grid_uv.clone().detach()


    # Get the corresponding UV coordinates for the given XY locations
    def get_UV_locations(self, xy_locations):
        return xy_locations
    

    def efficient_warping(self, ip1, pred_dxy):

        padded_ip1 = F.pad(ip1, (self.MSS, self.MSS, self.MSS, self.MSS), "constant", 0)
        
        # Scale and convert pred_dxy to long
        pred_dxy = (pred_dxy * (self.simulation_size / 2)).long()
        
        # Compute per-batch start coords
        start_x = self.MSS + pred_dxy[:, 0, 1]  # shape (B,)
        start_y = self.MSS + pred_dxy[:, 0, 0]  # shape (B,)

        B = padded_ip1.shape[0]
        
        X = torch.arange(self.simulation_size, device=self.device)
        Y = torch.arange(self.simulation_size, device=self.device)
        
        grid_x = X.view(1, -1, 1).expand(B, -1, self.simulation_size) + start_x.view(B, 1, 1)
        grid_y = Y.view(1, 1, -1).expand(B, self.simulation_size, -1) + start_y.view(B, 1, 1)

        b_idx = torch.arange(B, device=self.device).view(B, 1, 1)  # shape (B,1,1)
        ip2_pred = padded_ip1[b_idx, :, grid_x, grid_y].permute(0,3,1,2)

        mask_gathered = self.padded_mask[:, grid_x, grid_y]
        mask2 = mask_gathered.permute(1, 0, 2, 3).contiguous()

        return ip2_pred, mask2