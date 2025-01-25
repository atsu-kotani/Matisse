import comet_ml
import os
import torch
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from root_config import *
from Experiment.helper import largest_valid_region_square
from Experiment.ProgressLogger.Abstract import ProgressLogger
from Experiment.ProgressLogger import register_class

from matplotlib import font_manager
import matplotlib.pyplot as plt
font_manager.fontManager.addfont(f"{ROOT_DIR}/Tutorials/data/Futura.ttc")
prop = font_manager.FontProperties(fname=f"{ROOT_DIR}/Tutorials/data/Futura.ttc")
plt.rcParams['font.family'] = 'sans-serif'


@register_class("Local")
class Local(ProgressLogger):
    def __init__(self, experiment_name, required_image_resolution):
        super(Local, self).__init__(experiment_name, required_image_resolution)

        # load all PNG images in ProgressLogger/test_images, and reshape them to dim_imagexdim_imagex3
        test_images = []
        for file in os.listdir('Experiment/ProgressLogger/test_images'):
            if file.endswith('.png'):
                image = Image.open(f'Experiment/ProgressLogger/test_images/{file}').convert('RGB')
                image = image.resize((required_image_resolution, required_image_resolution))
            test_images.append(np.array(image) / 255.0)
        self.test_images = np.asarray(test_images)

        # save it as (BS, 3, dim_image, dim_image) tensor
        self.test_sRGB_images_full_field = torch.FloatTensor(self.test_images).to(DEVICE)

        self.experiment_name = experiment_name
        os.makedirs(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}', exist_ok=True)
        for i in range(self.test_sRGB_images_full_field.shape[0]):
            os.makedirs(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{i}', exist_ok=True)

        self.num_gradient_updates_list = []
        self.main_loss_list = []
        self.ns_cm_list = []
        self.ns_ip_list = []


    def log_progress(self,  simulating_tetra, retina, cortex, num_gradient_updates, 
                            main_loss, ns_cm_loss, ns_ip_loss, retina_spatial_sampling):

        # Padding for the images
        self.P = retina.LateralInhibition.get_kernel_size() // 2

        # Plotting the loss
        self.num_gradient_updates_list.append(num_gradient_updates)
        self.main_loss_list.append(main_loss.cpu().detach().numpy())

        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.num_gradient_updates_list, self.main_loss_list, label='Main Loss')
        plt.yscale('log')
        plt.xlabel('Number of Gradient Updates')
        plt.ylabel('Main Loss')
        plt.title('Main Loss')
        plt.savefig(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/main_loss.png')
        plt.close()

        if not simulating_tetra:
            # simulate the retina and cortex for the test images
            with torch.no_grad():
                test_batch_linsRGB = retina.CST.sRGB_to_linsRGB(self.test_sRGB_images_full_field)
                test_batch_LMS = retina.CST.linsRGB_to_LMS(test_batch_linsRGB).permute(0,3,1,2).unsqueeze(1) # shape: (BS, 1, 3, H, W)

                # Retina processing for the test images
                test_batch_warped_LMS_current_FoV = retina.SpatialSampling.forward(test_batch_LMS)
                test_batch_pa = retina.SpectralSampling.forward(test_batch_warped_LMS_current_FoV)
                test_batch_ons = retina.LateralInhibition.forward(test_batch_pa)

                # Cortical processing for the test images
                warped_ip1 = cortex.decode(test_batch_ons) # shape: (BS, 8, H, W)
                warped_ip1_linsRGB = cortex.ns_ip(warped_ip1) # shape: (BS, 3, H, W)
                warped_ip1_sRGB = retina.CST.linsRGB_to_sRGB(warped_ip1_linsRGB.permute(0,2,3,1)).permute(0,3,1,2) # shape: (BS, 3, H, W)

                if retina_spatial_sampling != 'Identity':
                    # Unwarp the image to the full field
                    internal_percept_sRGB, invalid_regions = cortex.get_unwarped_percept(warped_ip1_sRGB)
                    best_coords = largest_valid_region_square(invalid_regions)
                    pred_internal_percept_sRGB = internal_percept_sRGB[:, best_coords[0]:best_coords[2], best_coords[1]:best_coords[3]]
                else:
                    pred_internal_percept_sRGB = warped_ip1_sRGB.permute(0,2,3,1).cpu().detach().numpy()

            pred_internal_percept_sRGB = np.clip(pred_internal_percept_sRGB, 0, 1)

            # Plotting the internal percept
            for image_id, image in enumerate(pred_internal_percept_sRGB):

                # 2 images side by side
                fig, ax = plt.subplots(1,2, figsize=(10,5))
                ax[0].imshow(self.test_images[image_id])
                ax[0].set_title(f'Original')
                ax[0].axis('off')
                ax[1].imshow(image)
                ax[1].set_title(f'Predicted')
                ax[1].axis('off')

                fig.suptitle(f'After {num_gradient_updates:06d} gradient updates')

                # save at two locations
                plt.savefig(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}_current.png')
                plt.savefig(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}/{num_gradient_updates}.png')
                plt.close()


    def generate_progress_video(self):

        for image_id in range(self.test_images.shape[0]):
            # generate a gif from the images from the list of images
            images = []
            for gradient_update in range(100001):
                if os.path.exists(f"{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}/{gradient_update}.png"):
                    image = Image.open(f"{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}/{gradient_update}.png")
                    images.append(image)

            # save the gif from the list of images
            # imageio.mimsave(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}.gif', images, fps=20, loop=0)
            imageio.mimsave(f'{ROOT_DIR}/Experiment/Logging/{self.experiment_name}/IP{image_id}.gif', images, fps=20)
