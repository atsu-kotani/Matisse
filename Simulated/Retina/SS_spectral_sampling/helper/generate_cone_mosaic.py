#%%
import numpy as np
from PIL import Image


def generate_default_cone_mosaic(params):
    ons_dim = params['Experiment']['simulation_size']

    # get the cone types
    cone_types = params['RetinaModel']['retina_spectral_sampling']['cone_types']

    if cone_types == 'LMS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

    elif cone_types == 'L':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        cone_mosaic[:,:,0] = 1

    elif cone_types == 'M':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        cone_mosaic[:,:,1] = 1

    elif cone_types == 'S':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        cone_mosaic[:,:,2] = 1

    elif cone_types == 'LM':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        np.random.seed(3)
        # splitting S cones into L and M cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,2] == 1:
                    cone_mosaic[i,j,2] = 0
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 1
                    else:
                        cone_mosaic[i,j,1] = 1

    elif cone_types == 'LS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        # converting all M cones to L cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,1] == 1:
                    cone_mosaic[i,j,1] = 0
                    cone_mosaic[i,j,0] = 1

    elif cone_types == 'MS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        # converting all L cones to M cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    cone_mosaic[i,j,0] = 0
                    cone_mosaic[i,j,1] = 1

    elif cone_types == 'S+M_coexpress_S':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    cone_mosaic[i,j,1] = 0.5
                    cone_mosaic[i,j,2] = 0.5

    elif cone_types == 'S+M_ratio_S':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    r = np.random.random()
                    cone_mosaic[i,j,1] = r
                    cone_mosaic[i,j,2] = 1 - r

    elif cone_types == 'S+M_replace_S':
        cone_mosaic = np.zeros([ons_dim, ons_dim, 4])
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    cone_mosaic[i,j,1] = 1
                    cone_mosaic[i,j,2] = 0

    elif cone_types == 'MS+L_coexpress_M':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        # converting all L cones to M cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    cone_mosaic[i,j,0] = 0
                    cone_mosaic[i,j,1] = 1
        
        # converting MS cone mosaic to MS+L_coexpress_M
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,1] == 1:
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 0.5
                        cone_mosaic[i,j,1] = 0.5

    elif cone_types == 'MS+L_ratio_M':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        # converting all L cones to M cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    cone_mosaic[i,j,0] = 0
                    cone_mosaic[i,j,1] = 1
        
        # converting MS cone mosaic to MS+L_ratio_M
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,1] == 1:
                    if np.random.random() > 0.5:
                        r = np.random.random()
                        cone_mosaic[i,j,0] = r
                        cone_mosaic[i,j,1] = 1 - r

    elif cone_types == 'MS+L_replace_M':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)
        # converting all L cones to M cones
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    cone_mosaic[i,j,0] = 0
                    cone_mosaic[i,j,1] = 1
        
        # converting MS cone mosaic to MS+L_replace_M
        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,1] == 1:
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 1
                        cone_mosaic[i,j,1] = 0

    elif cone_types == 'LMSQ':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 0
                        cone_mosaic[i,j,3] = 1

    elif cone_types == 'LMS+Q_replace_L':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 0
                        cone_mosaic[i,j,3] = 1

    elif cone_types == 'LMS+Q_coexpress_L':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    if np.random.random() > 0.5:
                        cone_mosaic[i,j,0] = 0.5
                        cone_mosaic[i,j,3] = 0.5

    elif cone_types == 'LMS+Q_ratio_L':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if cone_mosaic[i,j,0] == 1:
                    if np.random.random() > 0.5:
                        r = np.random.random()
                        cone_mosaic[i,j,0] = r
                        cone_mosaic[i,j,3] = 1 - r
    
    elif cone_types == 'LMS+Q_replace_LMS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    cid = np.argmax(cone_mosaic[i,j])
                    cone_mosaic[i,j,cid] = 0
                    cone_mosaic[i,j,3] = 1
    
    elif cone_types == 'LMS+Q_ratio_LMS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    cid = np.argmax(cone_mosaic[i,j])
                    r = np.random.random()
                    cone_mosaic[i,j,cid] = r
                    cone_mosaic[i,j,3] = 1 - r
    
    elif cone_types == 'LMS+Q_coexpress_LMS':
        cone_mosaic = generate_default_LMS_cone_mosaic(ons_dim)

        np.random.seed(3)
        for i in range(ons_dim):
            for j in range(ons_dim):
                if np.random.random() > 0.5:
                    cid = np.argmax(cone_mosaic[i,j])
                    cone_mosaic[i,j,cid] = 0.5
                    cone_mosaic[i,j,3] = 0.5

    else:
        raise ValueError(f'Invalid cone types: {cone_types}')

    return cone_mosaic


def generate_default_LMS_cone_mosaic(ons_dim, L_ratio=1.92, M_ratio=1.0):
    
    np.random.seed(0)
    choice = [i for i in range(3)]
    ratios = (L_ratio / (L_ratio + M_ratio), M_ratio / (L_ratio + M_ratio), 0.0)
    cone_mosaic = np.eye(3)[np.random.choice(choice, size=(ons_dim, ons_dim), p=ratios)]
    s = np.sum(cone_mosaic)
    r = np.sum(cone_mosaic.reshape([-1, 3]), 0)

    S_search_width = 3
 
    cells_x_y = []
    for i in range(len(cone_mosaic)):
        for j in range(len(cone_mosaic)):
            cells_x_y.append([i,j])

    # shuffle the cells
    np.random.shuffle(cells_x_y)

    # for each cell, find the region, defined by the square of side 2H, that does not contain any S cones
    # if the region does not contain any S cones, then add a S cone to the region
    # Note that the cell_x_y is shuffled, so we are adding S cones to the mosaic in a random manner
    for [i,j] in cells_x_y:
        sx = np.max([0, i - S_search_width])
        ex = np.min([len(cone_mosaic), i + S_search_width])
        sy = np.max([0, j - S_search_width])
        ey = np.min([len(cone_mosaic), j + S_search_width])
        # if the region does not contain any S cones, then add a S cone to the region
        if np.sum(cone_mosaic[sx:ex,sy:ey].reshape(-1, 3), 0)[2] == 0:
            cone_mosaic[i,j,0] = 0
            cone_mosaic[i,j,1] = 0
            cone_mosaic[i,j,2] = 1

    s = np.sum(cone_mosaic)
    r = np.sum(cone_mosaic.reshape([-1, 3]), 0)
    
    # print (f'Ratio - L: {int(r[0]*100000/s)/1000.0}, M: {int(r[1]*100000/s)/1000.0}, S: {int(r[2]*100000/s)/1000.0}')
    # assert (np.sum(cone_mosaic) == ons_dim*ons_dim)

    cone_mosaic = np.pad(cone_mosaic, ((0,0),(0,0),(0,1)), 'constant', constant_values=0)

    return cone_mosaic


