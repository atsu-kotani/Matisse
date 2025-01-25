import numpy as np
from root_config import *


def generate_model_name(params):

    model_name = params.current_cone_mosaic
    
    if params.fourth_cone_type != None:
        model_name += f'_{params.fourth_cone_type}'

    if params.previous_cone_mosaic != None:
        model_name += f'_dim_boost_{params.previous_cone_mosaic}'

    model_name += f'_{params.dataset_name}'
    
    if params.ons_dim != 256:
        model_name += f'_{params.ons_dim}'
    if params.latent_dim != 8:
        model_name += f'_ld_{params.latent_dim}'

    if params.retina_spatial_sampling != 'Default':
        model_name += f'_FV_{params.enable_foveation}'
    if params.retina_spectral_sampling != 'Default':
        model_name += f'_SS_{params.retina_spectral_sampling}'
    if params.retina_lateral_inhibition != 'Default':
        model_name += f'_LI_{params.retina_lateral_inhibition}'
    if params.retina_eye_motion != 'Default':
        model_name += f'_EM_{params.retina_eye_motion}'


    if params.cortex_learn_eye_motion != 'Default':
        model_name += f'_M_{params.cortex_learn_eye_motion}'
    if params.cortex_learn_spatial_sampling != 'Default':
        model_name += f'_P_{params.cortex_learn_spatial_sampling}'
    if params.cortex_learn_cone_spectral_type != 'Default':
        model_name += f'_C_{params.cortex_learn_cone_spectral_type}'
    if params.cortex_learn_demosaicing != 'Default':
        model_name += f'_D_{params.cortex_learn_demosaicing}'
    if params.cortex_learn_lateral_inhibition != 'Default':
        model_name += f'_W_{params.cortex_learn_lateral_inhibition}'

    return model_name


def get_previous_model_name(params):
    from argparse import Namespace
    d = vars(params).copy()
    d['current_cone_mosaic'] = d['previous_cone_mosaic']
    d['previous_cone_mosaic'] = None
    d['dataset_type'] = 'NTIRE'
    d['fourth_cone_type'] = None
    d['load_pretrain_timestep'] = 100000
    old_params = Namespace(**d)
    old_model_name = generate_model_name(old_params)
    return old_model_name


def load_existing_timestep(params, experiment_name, cortex, main_optimizer, ns_cm_optimizer, ns_ip_optimizer, logging_timesteps):
    # Load existing timestep
    num_gradient_updates = 0

    for timestep in logging_timesteps[::-1]:
        if os.path.exists(f'{ROOT_DIR}/LearnedWeights/{experiment_name}/{timestep}_stats.pt'):
            num_gradient_updates = timestep
            break
    if num_gradient_updates > 0:
        print (f'Load timestep: {num_gradient_updates}')
        cortex.load_state_dict(torch.load(f'{ROOT_DIR}/LearnedWeights/{experiment_name}/{num_gradient_updates}.pt', weights_only=True))
        if params.fourth_cone_type == None:
            [all, ns_cm, ns_ip] = torch.load(f'{ROOT_DIR}/LearnedWeights/{experiment_name}/{num_gradient_updates}_stats.pt', weights_only=True)
            ns_ip_optimizer.load_state_dict(ns_ip)
        else:
            [all, ns_cm] = torch.load(f'{ROOT_DIR}/LearnedWeights/{experiment_name}/{num_gradient_updates}_stats.pt', weights_only=True)
        main_optimizer.load_state_dict(all)
        ns_cm_optimizer.load_state_dict(ns_cm)
        print (f'Resuming timestep: {num_gradient_updates}')
    
    return cortex, main_optimizer, ns_cm_optimizer, ns_ip_optimizer, num_gradient_updates


def compute_required_image_resolution(xy):
    '''
    Input: xy: torch tensor of shape (1, 2, H, W)
               Each x, y has to be in range [-1, 1]
    Output: required_image_resolution: int
    '''
    
    xy[:,0] -= torch.min(xy[:,0])
    xy[:,1] -= torch.min(xy[:,1])
    xy[:,0] /= torch.max(xy[:,0])
    xy[:,1] /= torch.max(xy[:,1])
    
    du = torch.sqrt(torch.sum((xy[0,:,1:,:] - xy[0,:,:-1,:]) ** 2, dim=0))
    dv = torch.sqrt(torch.sum((xy[0,:,:,1:] - xy[0,:,:,:-1]) ** 2, dim=0))

    multiplier = 1 / torch.min(torch.mean(torch.sort(du.view(-1), descending=False)[0][:10]), torch.mean(torch.sort(dv.view(-1), descending=False)[0][:10]))
    required_image_resolution = multiplier

    # make it power of 2
    required_image_resolution = 2 ** int(np.ceil(np.log2(required_image_resolution.item())))

    return required_image_resolution


# find the minimum and maximum x, y indices of the valid region
def largest_valid_region_square(matrix):
    rows, cols = matrix.shape
    dp = np.zeros_like(matrix, dtype=int)

    max_size = 0
    bottom_right = (0, 0)

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                if i == 0 or j == 0:
                    dp[i, j] = 1  # Edges of the matrix
                else:
                    dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + 1

                # Update maximum square size and position
                if dp[i, j] > max_size:
                    max_size = dp[i, j]
                    bottom_right = (i, j)

    # Calculate the top-left and bottom-right corners of the largest square
    y_max, x_max = bottom_right
    y_min, x_min = y_max - max_size + 1, x_max - max_size + 1

    return (y_min, x_min, y_max, x_max)