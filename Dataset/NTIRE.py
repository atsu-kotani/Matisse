import os
import torch
import numpy as np
from root_config import *
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from concurrent.futures import ThreadPoolExecutor

from Dataset.Abstract import Dataset
from Dataset import register_class


@register_class("NTIRE")
class NTIRE(Dataset):
    def __init__(self, params, retina):
        super(NTIRE, self).__init__(params, retina)

        dim_image = retina.required_image_resolution + (params['Experiment']['timesteps_per_image'] - 1) * 2 * params['RetinaModel']['max_shift_size']

        self.dataset_name = params['Dataset']['dataset_name']
        
        dataset_type = 'LMS'

        # if the peak frequencies are not the default values, add them to the dataset type (+ regenerate the dataset)
        if 'cone_fundamentals' in params['RetinaModel']['retina_spectral_sampling']:
            cone_fundamentals_params = params['RetinaModel']['retina_spectral_sampling']['cone_fundamentals']
        else:
            cone_fundamentals_params = {'L': 560, 'M': 530, 'S': 419}

        for key in cone_fundamentals_params:
            if key == 'L':
                L_peak = cone_fundamentals_params[key]
                if L_peak != 560: # default peak of the L-cone at 560 nm
                    dataset_type += f'_L{L_peak}'
            elif key == 'M':
                M_peak = cone_fundamentals_params[key]
                if M_peak != 530: # default peak of the M-cone at 530 nm
                    dataset_type += f'_M{M_peak}'
            elif key == 'S':
                S_peak = cone_fundamentals_params[key]
                if S_peak != 419: # default peak of the S-cone at 419 nm
                    dataset_type += f'_S{S_peak}'
            elif key == 'Q':
                Q_peak = cone_fundamentals_params[key]
                dataset_type += f'_Q{Q_peak}'

        if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/0899.pt'):
            print ('=== Spectrally interpolating NTIRE hyperspectral data... ===')
            interpolate_NTIRE_hyperspectral_data()
            print ('=== Done! ===')

        if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/{DATASET_SIZE-1}.pt'):
            print ('=== Preprocessing NTIRE hyperspectral data... ===')
            preprocess_NTIRE_hyperspectral_data(dim_image, dataset_type, retina.CST)
            print ('=== Done! ===')

        self.all_data = DatasetFolder(root=f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS', loader=self.loader, extensions='.pt')


    def __getitem__(self, index):
        index = index % len(self.all_data)
        data, _ = self.all_data[index]
        return data
        

    def __len__(self):
        return len(self.all_data)
    

    def loader(self, path):
        return torch.load(path, map_location='cpu', weights_only=True)


def interpolate_NTIRE_hyperspectral_data():
    import h5py

    os.makedirs(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data', exist_ok=True)
    data_folder = f'{ROOT_DIR}/Dataset/NTIRE2022_original/data'

    def interpolate(index):
        if os.path.exists(f'{data_folder}/ARAD_1K_{index:04d}.mat'):
            if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt'):
                # load mat file
                mat_contents = h5py.File(f'{data_folder}/ARAD_1K_{index:04d}.mat', 'r')
                
                bands = np.asarray(mat_contents['bands'])[:,0]
                cube = np.asarray(mat_contents['cube'])

                new_cube = []
                for i in range(len(bands)-1):
                    for j in range(10):
                        image1 = cube[i]
                        image2 = cube[i+1]
                        new_image = image1 * (1-j/10) + image2 * (j/10)
                        new_cube.append(new_image)
                
                new_cube.append(cube[-1])
                new_cube = np.asarray(new_cube)
                new_cube = torch.FloatTensor(new_cube)
                new_cube = new_cube.permute(2,1,0)

                torch.save(new_cube, f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt')


    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(interpolate, i) for i in range(1, 1001)]
    print ('Done!')


def preprocess_NTIRE_hyperspectral_data(dim_image, dataset_type, CST):

    defined_white_point = CST.white_point
    device = CST.device

    os.makedirs(f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/', exist_ok=True)
    os.makedirs(f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    def local_loader(path):
        return torch.load(path, weights_only=True, map_location=device)
    
    all_data = DatasetFolder(root=f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated', loader=local_loader, extensions='.pt')
    L = len(all_data)
        
    # global hyperspectral_cube_to_lms_rgb
    def hyperspectral_cube_to_lms(index):
        if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.pt'):
            cube = all_data[index][0]
            cube = cube.to(device)

            lms = torch.matmul(cube, CST.cone_fundamentals)
            
            (H, W, _) = lms.shape
            if H < dim_image or W < dim_image:
                if H < W:
                    multiplier = dim_image / H
                else:
                    multiplier = dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

                lms = F.interpolate((lms).permute(2,0,1).unsqueeze(0), size=(nH, nW), mode='bilinear', align_corners=False).squeeze().permute(1,2,0)

            # white world white balance
            current_white_point = (lms.reshape(-1, lms.shape[-1]).max(0)[0] + 1e-10) # (8, 3)
            lms = lms / current_white_point[None,None,:]
            lms *= defined_white_point

            lms_save = lms.detach().cpu()
            torch.save(lms_save, f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.pt')

    print ('First, converting hyperspectral data to LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(hyperspectral_cube_to_lms, i) for i in range(L)]
    print ('Done!')


    def crop_image(index):
        if index % 1000 == 0:
            print (f'{index} / {DATASET_SIZE}')
        i = index % L
        lms = torch.load(f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{i}.pt', weights_only=True)

        W, H = lms.shape[0], lms.shape[1]
        if W > dim_image:
            if H > dim_image:
                x = np.random.randint(0, W - dim_image)
                y = np.random.randint(0, H - dim_image)
            elif H == dim_image:
                x = np.random.randint(0, W - dim_image)
                y = 0
            else:
                print ('Hyperspectral image smaller than the required image resolution')
                raise ValueError
        elif W == dim_image:
            if H > dim_image:
                x = 0
                y = np.random.randint(0, H - dim_image)
            elif H == dim_image:
                x = 0
                y = 0
            else:
                print ('Hyperspectral image smaller than the required image resolution')
                raise ValueError

        lms = lms[x:x+dim_image, y:y+dim_image]
        torch.save(lms, f'{ROOT_DIR}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/{index}.pt')


    print ('Next, cropping the images...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(crop_image, i) for i in range(DATASET_SIZE)]
    print ('Done!')

