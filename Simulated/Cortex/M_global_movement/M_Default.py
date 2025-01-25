import numpy as np
import torch.nn.functional as F
from Experiment.helper import *
import torchvision.transforms as transforms
from Simulated.Cortex.M_global_movement.M_Abstract import AbstractGlobalMovement
from . import register_class

@register_class("Default")
class DefaultGlobalMovement(AbstractGlobalMovement):

    def __init__(self, params, device):
        super(DefaultGlobalMovement, self).__init__(params, device)

        self.device = device
        self.gaussian_blur_function = transforms.GaussianBlur(9, sigma=2.0)

        # Pre-compute constants
        self.shift_range = range(-2, 3)
        self.shifts = [(dx, dy) for dx in self.shift_range for dy in self.shift_range]


    def gaussian_blur(self, ons):
        # pad the ons by 4 pixels by reflecting the edges
        ons = F.pad(ons, (4, 4, 4, 4), mode='reflect')
        ons = self.gaussian_blur_function(ons)
        ons = ons[:, :, 4:-4, 4:-4]
        return ons


    def generate_grid_fixed(self, top_left, bottom_left, top_right, bottom_right, required_image_resolution):
        # Convert corner points to tensors and reshape for broadcasting
        tl = top_left.reshape(1, 1, 2)
        bl = bottom_left.reshape(1, 1, 2)
        tr = top_right.reshape(1, 1, 2)
        br = bottom_right.reshape(1, 1, 2)
        
        # Generate a meshgrid for the target image resolution and reshape for broadcasting
        y_coords, x_coords = torch.meshgrid(torch.linspace(0, 1, required_image_resolution), torch.linspace(0, 1, required_image_resolution), indexing='ij')
        y_coords = y_coords.unsqueeze(-1).to(self.device)  # Add an extra dimension for broadcasting
        x_coords = x_coords.unsqueeze(-1).to(self.device)  # Add an extra dimension for broadcasting
        
        # Calculate weights for bilinear interpolation
        top_weights = 1 - y_coords  # Weights for top edge (linearly decrease from top to bottom)
        bottom_weights = y_coords   # Weights for bottom edge (linearly increase from top to bottom)
        
        # Perform bilinear interpolation
        # Interpolate along the top and bottom edges
        top_interpolated = (1 - x_coords) * tl + x_coords * tr
        bottom_interpolated = (1 - x_coords) * bl + x_coords * br
        
        # Interpolate between the top and bottom interpolated points
        grid = top_weights * top_interpolated + bottom_weights * bottom_interpolated
        
        return grid
    
    
    def forward(self, ons1, ons2, P_cell_position, true_dxy=None):
        
        with torch.no_grad():
            ons1_ = ons1.detach().clone()
            ons2_ = ons2.detach().clone()

            # gaussian blur ons
            blurred_ons1 = self.gaussian_blur(ons1_)
            blurred_ons2 = self.gaussian_blur(ons2_)
            
            # unwarp the blurred ons
            xy_full = P_cell_position.get_XY_default_locations()
            required_image_resolution = compute_required_image_resolution(xy_full.detach().clone())
            
            grid = self.generate_grid_fixed(xy_full[0,:,0,0], xy_full[0,:,-1,0], xy_full[0,:,0,-1], xy_full[0,:,-1,-1], required_image_resolution)
            uvs = P_cell_position.get_UV_locations(grid.permute(2,0,1).unsqueeze(0))
            uvs = uvs.repeat(len(blurred_ons1), 1,1,1)

            full_blurred_ons1 = F.grid_sample(blurred_ons1, uvs.permute(0,2,3,1), align_corners=True, mode='bilinear', padding_mode='zeros')
            full_blurred_ons2 = F.grid_sample(blurred_ons2, uvs.permute(0,2,3,1), align_corners=True, mode='bilinear', padding_mode='zeros')
            mask = torch.ones_like(blurred_ons1).to(self.device)
            full_mask = F.grid_sample(mask, uvs.permute(0,2,3,1), align_corners=True, mode='nearest', padding_mode='zeros')

            # generate the image pyramid
            levels = int(np.log2(required_image_resolution)) - 2
            pyramid_blurred_ons1 = [full_blurred_ons1]
            pyramid_blurred_ons2 = [full_blurred_ons2]
            pyramid_mask = [full_mask]

            for i in range(levels):
                pyramid_blurred_ons1.append(F.avg_pool2d(pyramid_blurred_ons1[-1], 2, 2))
                pyramid_blurred_ons2.append(F.avg_pool2d(pyramid_blurred_ons2[-1], 2, 2))
                pyramid_mask.append(F.max_pool2d(pyramid_mask[-1], 2, 2))
                
            pyramid_blurred_ons1 = pyramid_blurred_ons1[::-1]
            pyramid_blurred_ons2 = pyramid_blurred_ons2[::-1]
            pyramid_mask = pyramid_mask[::-1]

            aggregate_shifts = torch.zeros(len(full_blurred_ons1), 2, device=self.device)
            for i in range(levels+1):
                # start with 4x4 images (8, 1, 4, 4)
                current_blurred_ons1 = pyramid_blurred_ons1[i]
                current_blurred_ons2 = pyramid_blurred_ons2[i]
                current_mask = pyramid_mask[i]
                current_res = current_blurred_ons1.shape[2]
                
                padded_current_blurred_ons1 = F.pad(current_blurred_ons1, (2, 2, 2, 2), mode='constant', value=0)
                padded_current_mask = F.pad(current_mask, (2, 2, 2, 2), mode='constant', value=0)

                shifted_ons1_list = [padded_current_blurred_ons1[:, :, dx+2:dx+2+current_res, dy+2:dy+2+current_res] for dx, dy in self.shifts]
                shifted_mask_list = [padded_current_mask[:, :, dx+2:dx+2+current_res, dy+2:dy+2+current_res] for dx, dy in self.shifts]
                
                ncc_scores = torch.ones(len(current_blurred_ons1), len(self.shifts), device=self.device) * 1000
                
                for j, (shifted_ons1, shifted_mask) in enumerate(zip(shifted_ons1_list, shifted_mask_list)):
                    error = torch.sum(torch.sqrt((shifted_ons1 - current_blurred_ons2) ** 2) * (shifted_mask * current_mask), dim=(2, 3)) / (torch.sum(shifted_mask * current_mask, dim=(2, 3)) + 1e-6)
                    ncc_scores[:, j] = error.squeeze()
                    
                _, max_indices = torch.min(ncc_scores, 1)
                dx = max_indices % len(self.shift_range) - 2
                dy = max_indices // len(self.shift_range) - 2
                aggregate_shifts += torch.stack([dx, dy], -1) * (2 ** (levels - i))
                
                # shift all images of the upper layers in the pyramid
                for j in range(i+1, len(pyramid_blurred_ons1)):
                    cdx = dy * 2**(j - i)
                    cdy = dx * 2**(j - i)
                    P = 2 ** (j - i) * 2

                    new_blurred_ons1 = F.pad(pyramid_blurred_ons1[j], (P,P,P,P), mode='constant', value=0)
                    current_res = pyramid_blurred_ons1[j].shape[2]
                    new_blurred_ons1 = [new_blurred_ons1[k,:,cdx[k]+P:cdx[k]+P+current_res,cdy[k]+P:cdy[k]+P+current_res] for k in range(len(new_blurred_ons1))]
                    pyramid_blurred_ons1[j] = torch.stack(new_blurred_ons1)
                    new_mask = F.pad(pyramid_mask[j], (P,P,P,P), mode='constant', value=0)
                    new_mask = [new_mask[k,:,cdx[k]+P:cdx[k]+P+current_res,cdy[k]+P:cdy[k]+P+current_res] for k in range(len(new_mask))]
                    pyramid_mask[j] = torch.stack(new_mask)

            pred_dxy = aggregate_shifts.reshape([-1, 2]) / (required_image_resolution / 2)
            pred_dxy = pred_dxy.detach().unsqueeze(1)
            
        return pred_dxy
        