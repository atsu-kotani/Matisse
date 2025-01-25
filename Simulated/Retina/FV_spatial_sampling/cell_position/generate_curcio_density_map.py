#%%
import pickle
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from curcio_density_data import *
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

from root_config import ROOT_DIR

#%%
font_manager.fontManager.addfont(f"{ROOT_DIR}/Assets/fonts/GillSans.ttc")
prop = font_manager.FontProperties(fname=f"{ROOT_DIR}/Assets/fonts/GillSans.ttc")
plt.rcParams["font.size"] = 50
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['figure.autolayout'] = True

nasal = np.asarray(nasal)
temporal = np.asarray(temporal)
superior = np.asarray(superior)
inferior = np.asarray(inferior)

nasal_eq = interpolate.interp1d(temporal[:,0], temporal[:,1] * 1000, kind='cubic')
tempo_eq = nasal_eq
super_eq = nasal_eq
infer_eq = nasal_eq

ret_size = 4000

def generate_human_cone_densitiy_map():
    center = ret_size // 2
    cell_spacing = np.zeros([ret_size, ret_size])

    nasals = [nasal_eq(i/1000) for i in range(ret_size*2)]
    tempos = [tempo_eq(i/1000) for i in range(ret_size*2)]
    supers = [super_eq(i/1000) for i in range(ret_size*2)]
    infers = [infer_eq(i/1000) for i in range(ret_size*2)]

    for i in range(ret_size):
        print (i)
        for j in range(ret_size):
                x = i - center
                y = j - center
                d = int(np.round(np.sqrt(x**2 + y**2)))
                if x > 0:
                    if y > 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = nasals[d] * (r) + supers[d] * (1-r)
                    elif y < 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = nasals[d] * (1-r) + infers[d] * (r)
                    else:
                        cell_spacing[i,j] = nasals[d]
                elif x < 0:
                    if y > 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = tempos[d] * (1-r) + supers[d] * (r)
                    elif y < 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = tempos[d] * r + infers[d] * (1-r)
                    else:
                        cell_spacing[i,j] = tempos[d]
                else:
                    if y > 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = supers[d]
                    elif y < 0:
                        r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                        cell_spacing[i,j] = infers[d]
                    else:
                        cell_spacing[i,j] = tempos[0]

    with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_human_{ret_size}.cpkl', 'wb') as f:
        pickle.dump(cell_spacing, f)

    fmt = lambda x, pos: "%d"%(x // 1000)
    fig = plt.figure(figsize=(20,20))
    plt.imshow(cell_spacing, cmap='bone')
    cbar = plt.colorbar(fraction=0.046, pad=0.04, format=FuncFormatter(fmt), label='Cone Cell Density / 1000')
    plt.xlabel('Eccentricity ($\mu m$)')
    plt.savefig(f'{ROOT_DIR}/Assets/cell_position/data/curcio_human_{ret_size}.png')
    plt.close()


def generate_horizontal_streak_cone_densitiy_map():
    with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_human_{ret_size}.cpkl', 'rb') as f:
        cell_spacing = pickle.load(f)

    cell_spacing = np.repeat(cell_spacing[:,ret_size//2:ret_size//2+1], ret_size, 1)

    am = np.min(cell_spacing)
    ama = np.max(cell_spacing)
    cell_spacing = cell_spacing - am
    cell_spacing = cell_spacing ** 2
    cell_spacing /= np.max(cell_spacing)
    cell_spacing = cell_spacing * (ama-am) + am * 2

    fmt = lambda x, pos: "%d"%(x // 1000)
    fig = plt.figure(figsize=(20,20))
    plt.imshow(cell_spacing, cmap='bone')
    cbar = plt.colorbar(fraction=0.046, pad=0.04, format=FuncFormatter(fmt), label='Cone Cell Density / 1000')
    plt.xlabel('Eccentricity ($\mu m$)')
    plt.savefig(f'{ROOT_DIR}/Assets/cell_position/data/curcio_hor_streak_{ret_size}.png')
    plt.close()

    with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_hor_streak_{ret_size}.cpkl', 'wb') as f:
        pickle.dump(cell_spacing, f)


def generate_two_fovea_cone_densitiy_map():

    center_x = ret_size // 2
    center_y = ret_size // 2.1
    cell_spacing = np.zeros([ret_size, ret_size])

    nasals = [nasal_eq(i/1000) for i in range(ret_size*2)]
    tempos = [tempo_eq(i/1000) for i in range(ret_size*2)]
    supers = [super_eq(i/1000) for i in range(ret_size*2)]
    infers = [infer_eq(i/1000) for i in range(ret_size*2)]

    for i in range(ret_size):
        print (i)
        for j in range(ret_size):
            # if j < ret_size//2:
            x = i - center_x
            y = j - center_y
            d = int(np.round(np.sqrt(x**2 + y**2)))
            if x > 0:
                if y > 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = nasals[d] * (r) + supers[d] * (1-r)
                elif y < 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = nasals[d] * (1-r) + infers[d] * (r)
                else:
                    cell_spacing[i,j] = nasals[d]
            elif x < 0:
                if y > 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = tempos[d] * (1-r) + supers[d] * (r)
                elif y < 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = tempos[d] * r + infers[d] * (1-r)
                else:
                    cell_spacing[i,j] = tempos[d]
            else:
                if y > 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = supers[d]
                elif y < 0:
                    r = np.arctan(x / y) / np.pi * 180.0 % 90.0 / 90.0
                    cell_spacing[i,j] = infers[d]
                else:
                    cell_spacing[i,j] = tempos[0]

    a = cell_spacing + np.flip(cell_spacing, 1)
    a = cell_spacing[:,:ret_size//2]
    b = np.flip(a, 1)
    a = np.concatenate([a,b], 1)

    am = np.min(cell_spacing)
    ama = np.max(cell_spacing)

    cell_spacing = cell_spacing - am
    cell_spacing = cell_spacing ** 2
    cell_spacing /= np.max(cell_spacing)
    cell_spacing = cell_spacing * (ama-am) + am * 2

    cell_spacing = cell_spacing + np.flip(cell_spacing, 1)
    cell_spacing -= np.min(cell_spacing)
    cell_spacing /= np.max(cell_spacing)
    cell_spacing = cell_spacing * (ama-am) + am
    
    with open(f'{ROOT_DIR}/Assets/cell_position/data/curcio_two_fovea_{ret_size}.cpkl', 'wb') as f:
        pickle.dump(cell_spacing, f)

    fmt = lambda x, pos: "%d"%(x // 1000)
    fig = plt.figure(figsize=(20,20))
    # cell_spacing = cell_spacing[400:-400,400:-400]
    plt.imshow(cell_spacing, cmap='bone')
    cbar = plt.colorbar(fraction=0.046, pad=0.04, format=FuncFormatter(fmt), label='Cone Cell Density / 1000')
    plt.xlabel('Eccentricity ($\mu m$)')
    plt.savefig(f'{ROOT_DIR}/Assets/cell_position/data/curcio_two_fovea_{ret_size}.png')
    plt.close()

# %%
generate_human_cone_densitiy_map()
generate_horizontal_streak_cone_densitiy_map()
generate_two_fovea_cone_densitiy_map()
# %%
