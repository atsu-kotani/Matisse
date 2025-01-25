import torch

import pickle    
# figuring out the path
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch.optim as optim
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from root_config import ROOT_DIR

from matplotlib import font_manager
font_manager.fontManager.addfont(f"{ROOT_DIR}/Assets/fonts/GillSans.ttc")
prop = font_manager.FontProperties(fname=f"{ROOT_DIR}/Assets/fonts/GillSans.ttc")
# plt.rcParams["font.size"] = 40
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

retsize = 4000

class Model(nn.Module):
    
    def __init__(self, ons_dim, subsample=8, data=None):
        super(Model, self).__init__()
        
        self.ons_dim = ons_dim
        if data is None:
            x_ons_dim = torch.FloatTensor(np.linspace(-self.ons_dim, self.ons_dim, self.ons_dim // subsample))
            y_ons_dim = torch.FloatTensor(np.linspace(-self.ons_dim, self.ons_dim, self.ons_dim // subsample))
            xx_ons_dim, yy_ons_dim = torch.meshgrid(x_ons_dim, y_ons_dim, indexing='xy')
            self.base_loc_ons_dim = torch.stack([xx_ons_dim, yy_ons_dim], 0).unsqueeze(0) * 0.8
            self.position = nn.Parameter(self.base_loc_ons_dim)
        else:
            self.position = nn.Parameter(data)
        
    def forward(self):
        return self.position.permute(0,2,3,1)


def main(params, device):

    np.random.seed(0)
    torch.manual_seed(0)

    N = params.ons_dim
    D = N + 32

    print (params.cone_loc_type)

    if params.cone_loc_type == 1:
        with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_human_{retsize}.cpkl', 'rb') as f:
            cell_position = pickle.load(f)
        data_name = 'human'
    elif params.cone_loc_type == 2:
        with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_hor_streak_{retsize}.cpkl', 'rb') as f:
            cell_position = pickle.load(f)
        data_name = 'hor_streak'
    elif params.cone_loc_type == 3:
        with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_two_fovea_{retsize}.cpkl', 'rb') as f:
            cell_position = pickle.load(f)
        data_name = 'two_fovea'
    else:
        raise ValueError(f'Invalid cone location type: {params.cone_loc_type}')

    im = np.asarray(cell_position)
    im = torch.FloatTensor(im).reshape(1,1,im.shape[0],im.shape[1]).to('cuda:0')
    cell_position = np.flip(cell_position, 0)

    mask = np.ones([1,1,9,D,D])
    mask[0,0,4,:,:] = 0
    mask = torch.FloatTensor(mask).to('cuda:0')

    c = np.zeros([N, N, 3])
    for i in range(N):
        for j in range(N):
            if (i + j) % 2 == 0:
                c[i,j,0] = 1
            else:
                c[i,j,2] = 1

    mul = 1.0
    tr = 100000000
    
    os.makedirs(f'{ROOT_DIR}/Assets/cell_position/cell_pos_evolution/{params.cone_loc_type}_{params.ons_dim}/', exist_ok=True)

    for subsample in [8, 4, 2, 1]:
        if subsample == 8:
            model = Model(D, subsample=subsample).to(device)
            model = torch.compile(model)
            params_list = [{'params': model.parameters()}]
            all_optimizer = optim.Adam(params_list, lr=0.01)
            all_optimizer.zero_grad(set_to_none=True)
        else:
            with open(f'{ROOT_DIR}/Assets/cell_position/cell_pos_evolution/{params.cone_loc_type}_{params.ons_dim}/50000.cpkl', 'rb') as f:
                [pos, density, cell_position] = pickle.load(f)
            pos = torch.FloatTensor(pos).to(device)
            pos = pos.permute(2,0,1)[:,::subsample,::subsample].unsqueeze(0)
            del model
            model = Model(ons_dim=D, subsample=subsample, data=pos).to(device)
            model = torch.compile(model)
            params_list = [{'params': model.parameters()}]
            all_optimizer = optim.Adam(params_list, lr=0.01)
            all_optimizer.zero_grad(set_to_none=True)

        for step in range(50001):
            pos = model()
            if subsample > 1:
                pos = F.interpolate(pos.permute(0,3,1,2), size=(D,D), mode='bilinear', align_corners=True).permute(0,2,3,1)

            pos_for_sampling_image = pos / (retsize//2)
            true_density = F.grid_sample(im, pos_for_sampling_image, padding_mode='border', align_corners=True)[0,0,1:-1,1:-1] # (1,1,128,128)
            
            pos_for_computing_area = pos.permute(0,3,1,2) / 1000
            pos_im_pad = F.pad(pos_for_computing_area, (1,1,1,1), mode='reflect')
            surround = F.unfold(pos_im_pad, (3,3)).reshape(1,2,9,D,D)[:,:,:,1:-1,1:-1]

            center = surround[:,:,4:5]

            surr8 = torch.cat([surround[:,:,:3], surround[:,:,5:6], surround[:,:,8:9], surround[:,:,7:8], surround[:,:,6:7], surround[:,:,3:4]], 2)

            # verx = (surr8 - center) / 2 + center
            verx = surr8
            di = len(verx[0,0])
            x = verx[0,0].reshape(di, -1)
            y = verx[0,1].reshape(di, -1)
            area = (0.5 * torch.abs(torch.sum(x*torch.roll(y,1,0),0) - torch.sum(y*torch.roll(x,1,0),0))).reshape([D-2,D-2]) + 1e-10
            density = 1 / area
            e1 = torch.mean(torch.abs(true_density - density))
            
            center_pred = torch.mean(surr8, 2, keepdim=True)
            e2 = torch.mean(torch.sqrt(torch.sum((center - center_pred)**2, 1)+1e-20)) * tr * mul

            loss =  e1 + e2
            loss.backward()
            all_optimizer.step()
            all_optimizer.zero_grad(set_to_none=True)

            if step % 10000 == 0:
                with torch.no_grad():
                    pos2 = model()
                    pos2 = F.interpolate(pos2.permute(0,3,1,2), size=(D,D), mode='bilinear', align_corners=True).permute(0,2,3,1)
                pos2 = pos2.cpu().detach().numpy()[0] #[0,1:-1,1:-1,:]
                with open(f'{ROOT_DIR}/Assets/cell_position/cell_pos_evolution/{params.cone_loc_type}_{params.ons_dim}/{step}.cpkl', 'wb') as f:
                    pickle.dump([pos2, density, cell_position], f)
            
            print (f'Step: {step}, \tLoss: {int(loss.item())} \tE1: {int(e1.item())} \tE2: {int(e2.item())}')
    
    pos = pos.cpu().detach().numpy() / 1000
    result = pos[0,16:-16,16:-16]
    print (result.shape)
    with open(f'{ROOT_DIR}/Assets/cell_position/data/cone_locs_{data_name}_{params.ons_dim}.cpkl', 'wb') as f:
        pickle.dump(result, f)
    
    fig = plt.figure(figsize=(50,50))    
    plt.scatter(result[:,:,0], result[:,:,1], s=0.4)
    plt.axis('equal')
    plt.savefig(f'{ROOT_DIR}/Assets/cell_position/data/res_{data_name}_{params.ons_dim}.png')
    plt.close()
    
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for cin in [1, 2, 3]:
        parser = argparse.ArgumentParser(description='Arguments for experiments')
        parser.add_argument('-od',  '--ons_dim', type=int, default=512)
        parser.add_argument('-cin',  '--cone_loc_type', type=int, default=cin)
        main(parser.parse_args(), device)

