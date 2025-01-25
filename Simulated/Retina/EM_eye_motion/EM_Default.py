import torch
from root_config import *
from Simulated.Retina.EM_eye_motion.EM_Abstract import AbstractEyeMotion
from . import register_class

@register_class("Default")
class DefaultEyeMotion(AbstractEyeMotion):
    def __init__(self, params, device):
        super(DefaultEyeMotion, self).__init__(params, device)

        # how many timesteps to sample per image
        self.timesteps_per_image = params['Experiment']['timesteps_per_image']
        self.MSS = params['RetinaModel']['max_shift_size']
        self.device = device
        

    def forward(self, LMS_full_field, required_image_resolution):
        (batch_size, _, H, W) = LMS_full_field.shape

        batch_true_dxy = []
        batch_LMS_current_FoV = []

        batch_true_dxy = torch.zeros((batch_size, self.timesteps_per_image-1, 2), device=self.device)
        batch_LMS_current_FoV = torch.zeros((batch_size, self.timesteps_per_image, 4, required_image_resolution, required_image_resolution), device=self.device)

        batch_LMS_current_FoV[:,0] = LMS_full_field[:,:,self.MSS:self.MSS+required_image_resolution,self.MSS:self.MSS+required_image_resolution]

        for i in range(batch_size):
            x, y = self.MSS, self.MSS

            for t in range(self.timesteps_per_image-1):
                dx, dy = torch.randint(-self.MSS, self.MSS, size=(2,))
                new_x, new_y = x+dx, y+dy
                new_x = torch.clamp(new_x, 0, W-required_image_resolution)
                new_y = torch.clamp(new_y, 0, H-required_image_resolution)
                dx, dy = new_x-x, new_y-y

                batch_true_dxy[i,t] = torch.FloatTensor([dx, dy]).to(self.device)
                batch_LMS_current_FoV[i,t+1] = LMS_full_field[i,:,new_y:new_y+required_image_resolution,new_x:new_x+required_image_resolution]
                x,y = new_x, new_y
            
        return batch_LMS_current_FoV, batch_true_dxy
