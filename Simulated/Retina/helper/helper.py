import math
import torch
import pickle
import numpy as np
from root_config import *
import torch.nn.functional as F
from Experiment.helper import *



def get_cone_sampling_map(params, cone_distribution_type='Human'):

    with open(f'{ROOT_DIR}/Simulated/Retina/FV_spatial_sampling/cell_position/data/cone_locs_{cone_distribution_type}_512.cpkl', 'rb') as f:
        cone_locs_in_ecc = pickle.load(f)

    (height, _, _) = cone_locs_in_ecc.shape

    D = params['Experiment']['simulation_size']
    N = height
    P = (height - D) // 2

    cone_locs_in_ecc /= np.max(np.abs(cone_locs_in_ecc))

    torch.manual_seed(0)
    noise_x = rand_perlin_2d_octaves((N, N), (2,2), 6, 0.8)
    noise_y = rand_perlin_2d_octaves((N, N), (2,2), 6, 0.8)

    noise = torch.stack([noise_x, noise_y], -1)
    noise = F.pad(noise.permute(2,0,1)[:,1:-1,1:-1], (1,1,1,1), 'constant').permute(1,2,0)

    x = np.linspace(0, N-1, N)
    y = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    base_loc = np.stack([xx,yy], -1)
    base_loc = torch.FloatTensor(base_loc)
    base_loc2 = base_loc + noise
    base_loc = ((base_loc / (N-1)) - 0.5) * 2
    base_loc2 = ((base_loc2 / (N-1)) - 0.5) * 2

    # without perlin noise
    B_ = base_loc.unsqueeze(0)
    A_ = torch.FloatTensor(cone_locs_in_ecc).permute(2,0,1).unsqueeze(0)
    cone_locs_in_ecc = F.grid_sample(A_, B_, align_corners=True)
    cone_locs_in_ecc = cone_locs_in_ecc[0].permute(1,2,0).cpu().detach().numpy()

    # with perlin noise
    B = base_loc2.unsqueeze(0)
    A = torch.FloatTensor(cone_locs_in_ecc).permute(2,0,1).unsqueeze(0)
    noisy_cone_locs_in_ecc = F.grid_sample(A, B, align_corners=True)
    noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc[0].permute(1,2,0).cpu().detach().numpy()

    if P > 0:
        noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc[P:-P,P:-P]
        cone_locs_in_ecc = cone_locs_in_ecc[P:-P,P:-P]

    noisy_cone_locs_in_ecc = (noisy_cone_locs_in_ecc - np.min(noisy_cone_locs_in_ecc))
    noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc / np.max(noisy_cone_locs_in_ecc)

    required_image_resolution = compute_required_image_resolution(torch.FloatTensor(noisy_cone_locs_in_ecc).unsqueeze(0).permute(0,3,1,2))

    cone_locs = (noisy_cone_locs_in_ecc - 0.5) * 2.0
    
    # computing mipmap_level
    tmp_cone_locs = noisy_cone_locs_in_ecc * required_image_resolution
    du = np.sqrt(np.sum((tmp_cone_locs[1:,:] - tmp_cone_locs[:-1,:])**2, -1))
    dv = np.sqrt(np.sum((tmp_cone_locs[:,1:] - tmp_cone_locs[:,:-1])**2, -1))
    du = np.pad(du, ((0,1), (0,0)), 'reflect')
    dv = np.pad(dv, ((0,0), (0,1)), 'reflect')
    mip_level = np.log2(np.max(np.stack([du, dv]), 0))
    
    return cone_locs, mip_level, required_image_resolution


# from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing='ij'), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise