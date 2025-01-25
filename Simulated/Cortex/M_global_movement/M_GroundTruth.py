from Experiment.helper import *
from Simulated.Cortex.M_global_movement.M_Abstract import AbstractGlobalMovement
from . import register_class

@register_class("GroundTruth")
class GroundTruthGlobalMovement(AbstractGlobalMovement):

    def __init__(self, params, device):
        super(GroundTruthGlobalMovement, self).__init__(params, device)
        self.device = device

    
    def forward(self, ons1, ons2, P_cell_position, true_dxy=None):
        
        return true_dxy


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
    