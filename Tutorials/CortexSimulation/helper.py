# Load root configuration
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', '..'))
from root_config import ROOT_DIR, DEVICE

# Matplotlib settings for later use in visualization
from matplotlib import font_manager
import matplotlib.pyplot as plt
font_manager.fontManager.addfont(f"{ROOT_DIR}/Tutorials/data/Futura.ttc")
prop = font_manager.FontProperties(fname=f"{ROOT_DIR}/Tutorials/data/Futura.ttc")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 20
plt.rcParams['font.sans-serif'] = prop.get_name()


# Pre-import all the necessary modules
import yaml
import torch
import IPython
import subprocess
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import imageio.v2 as imageio
import torch.nn.functional as F
from Experiment.helper import compute_required_image_resolution, largest_valid_region_square

# Pre-define the helper functions for later use

def load_sRGB_image(retina, image_path, params):
    # Loading the input stimulus
    MSS = params['RetinaModel']['max_shift_size']
    current_image_resolution = retina.required_image_resolution + MSS * 2

    image = Image.open(image_path).convert('RGB')
    H, W = image.size
    if H != W:
        if H > W:
            # crop the image
            image = image.crop((0, (H-W)//2, W, (H+W)//2))
        else:
            # crop the image
            image = image.crop(((W-H)//2, 0, (W+H)//2, H))

    image = image.resize((current_image_resolution, current_image_resolution), Image.Resampling.LANCZOS)
    image = np.asarray(image).copy() / 255.0
    image_tensor = torch.FloatTensor(image).to(DEVICE)
    return image_tensor


def get_unwarped_percept(warped_ip_sRGB, cortex):
    xy_full = cortex.P_cell_position.get_XY_default_locations()
    required_image_resolution = compute_required_image_resolution(xy_full.detach().clone())

    grid = cortex.M_global_movement.generate_grid_fixed(xy_full[0,:,0,0], xy_full[0,:,-1,0], xy_full[0,:,0,-1], xy_full[0,:,-1,-1], required_image_resolution)
    uvs = cortex.P_cell_position.get_UV_locations(grid.permute(2,0,1).unsqueeze(0))
    uvs = uvs.repeat(len(warped_ip_sRGB), 1,1,1)

    if cortex.device == 'mps:0':
        # F.grid_sample function acts strangely on MPS, so we move to CPU
        warped_ip_sRGB = warped_ip_sRGB.to('cpu')
        uvs = uvs.to('cpu')
        uvs = torch.clip(uvs, -1, 1)

    internal_percept_sRGB = F.grid_sample(warped_ip_sRGB, uvs.permute(0,2,3,1), align_corners=True, mode='bilinear', padding_mode='zeros')

    internal_percept_sRGB = internal_percept_sRGB.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    internal_percept_sRGB = np.clip(internal_percept_sRGB, 0, 1)
    invalid_regions = (((uvs <= -1) | (uvs >= 1)).sum(1) > 0).squeeze(0).cpu().detach().numpy()
    invalid_regions = invalid_regions.astype(int) # (512, 512) image

    best_coords = largest_valid_region_square(invalid_regions)
    cropped_internal_percept_sRGB = internal_percept_sRGB[best_coords[0]:best_coords[2], best_coords[1]:best_coords[3]]

    return cropped_internal_percept_sRGB
