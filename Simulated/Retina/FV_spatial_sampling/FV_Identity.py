from root_config import *
from Simulated.Retina.helper.helper import *
from Simulated.Retina.FV_spatial_sampling.FV_Abstract import AbstractSpatialSampling
from Simulated.Retina.FV_spatial_sampling import register_class

@register_class("Identity")
class IdentitySpatialSampling(AbstractSpatialSampling):
    def __init__(self, params, device):
        super(IdentitySpatialSampling, self).__init__(params, device)

        self.required_image_resolution = params['Experiment']['simulation_size']

        self.simulation_size = params['Experiment']['simulation_size']
        x_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        y_ons_dim = torch.FloatTensor(np.linspace(0, self.simulation_size-1, self.simulation_size))
        xx_ons_dim, yy_ons_dim = torch.meshgrid(x_ons_dim, y_ons_dim, indexing='xy')
        regular_grid = torch.stack([xx_ons_dim, yy_ons_dim], 0).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
        self.cone_locs = (regular_grid - ((self.simulation_size-1)/2)) / ((self.simulation_size-1)/2)
        self.cone_locs = self.cone_locs[0].permute(1,2,0)


    def forward(self, image):
        return image
    