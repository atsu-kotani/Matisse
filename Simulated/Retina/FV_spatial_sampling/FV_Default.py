import torch
from root_config import *
import torch.nn.functional as F
from Simulated.Retina.helper.helper import *
from Simulated.Retina.FV_spatial_sampling.FV_Abstract import AbstractSpatialSampling
from Simulated.Retina.FV_spatial_sampling import register_class

@register_class("Default")
class DefaultSpatialSampling(AbstractSpatialSampling):
    def __init__(self, params, device):
        super(DefaultSpatialSampling, self).__init__(params, device)

        self.device = device

        self.optic_nerve_signal_dim = params['Experiment']['simulation_size']
        self.max_shift_size = params['RetinaModel']['max_shift_size']

        # with foveation, base model
        cone_locs, mip_level, self.required_image_resolution = get_cone_sampling_map(params, cone_distribution_type='Human')

        self.cone_locs = torch.FloatTensor(cone_locs).to(device).detach()
        self.mip_level = torch.FloatTensor(mip_level).to(device).detach()

        self.grid = self.cone_locs.clone().unsqueeze(0)
        ll = torch.floor(self.mip_level).type(torch.LongTensor).to(device).detach()
        ul = torch.ceil(self.mip_level).type(torch.LongTensor).to(device).detach()
        self.max_mip_level = torch.max(ul) + 1

        self.du = self.mip_level - ll
        self.dl = ul - self.mip_level
        self.ll = F.one_hot(ll, num_classes=self.max_mip_level).permute(2,0,1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.ul = F.one_hot(ul, num_classes=self.max_mip_level).permute(2,0,1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.dl[self.dl + self.du != 1] = 1
        self.dl = self.dl.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.du = self.du.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        if self.device == 'mps:0':
            self.grid = self.grid.to('cpu')


    def forward(self, im0):
        (BS, N, C, Hm, Wm) = im0.shape 
        D = self.optic_nerve_signal_dim

        if self.device == 'mps:0':
            im0 = im0.to('cpu')

        mip = []
        DIV = 1
        for _ in range(self.max_mip_level):
            d1 = F.grid_sample(im0.reshape([BS*N,C,Hm//DIV,Wm//DIV]), self.grid.repeat(BS*N,1,1,1), align_corners=True, mode='bilinear').reshape(BS,N,C,D,D)
            mip.append(d1.clone())
            im0 = mipmap(im0.clone(), 2)
            DIV *= 2

        mip = torch.stack(mip, 0)
        if self.device == 'mps:0':
            mip = mip.to('mps:0')

        l = torch.sum(self.ll * mip, 0)
        u = torch.sum(self.ul * mip, 0)
        rgb1 = l * self.dl + u * self.du
        
        return rgb1
        
    

def mipmap(orig, N):
    (BS,D,C,H,W) = orig.shape
    nH = H // N
    nW = W // N
    return orig.reshape(BS,D,C,nH,N,nW,N).mean(6).mean(4)
