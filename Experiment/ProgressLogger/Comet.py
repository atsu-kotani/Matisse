import comet_ml
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from root_config import *
from Experiment.helper import largest_valid_region_square
from Experiment.ProgressLogger.Abstract import ProgressLogger
from Experiment.ProgressLogger import register_class


@register_class("Comet")
class Comet(ProgressLogger):
    def __init__(self, experiment_name, required_image_resolution):
        super(Comet, self).__init__(experiment_name, required_image_resolution)

        self.experiment = comet_ml.Experiment(project_name='Results', auto_metric_logging=LOG_SYSTEM_METRICS, log_env_gpu=LOG_SYSTEM_METRICS, log_env_host=LOG_SYSTEM_METRICS, log_env_cpu=LOG_SYSTEM_METRICS, log_env_disk=LOG_SYSTEM_METRICS, log_env_network=LOG_SYSTEM_METRICS)
        self.experiment.set_name(experiment_name)
        self.experiment.log_code(folder='Simulated/Cortex/')

        # load all PNG images in ProgressLogger/test_images, and reshape them to dim_imagexdim_imagex3
        test_images = []
        for file in os.listdir('Experiment/ProgressLogger/test_images'):
            if file.endswith('.png'):
                image = Image.open(f'Experiment/ProgressLogger/test_images/{file}').convert('RGB')
                image = image.resize((required_image_resolution, required_image_resolution))
            test_images.append(np.array(image) / 255.0)
        test_images = np.asarray(test_images)

        # save it as (BS, 3, dim_image, dim_image) tensor
        self.test_sRGB_images_full_field = torch.FloatTensor(test_images).to(DEVICE)


    def log_progress(self,  simulating_tetra, retina, cortex, num_gradient_updates, 
                            main_loss, ns_cm_loss, ns_ip_loss,
                            true_eye_movement, pred_eye_movement,
                            true_LI_kernel, pred_LI_kernel,
                            true_cone_mosaic_numpy, pred_cone_mosaic,
                            true_cone_locations, pred_cone_locations,
                            ons1, ons2, ons2_pred):

        # Padding for the images
        self.P = retina.LateralInhibition.get_kernel_size() // 2

        # Plotting the loss
        self.experiment.log_metric('Main Loss', main_loss, step=num_gradient_updates)
        self.experiment.log_metric('NS_CM Loss', ns_cm_loss, step=num_gradient_updates)

        # Plotting the eye movement (True and Predicted)
        self.plot_eye_movement_figure(pred_eye_movement, true_eye_movement, num_gradient_updates)
        # Plotting the cone locations (True and Predicted)
        self.plot_cone_locations_figure(pred_cone_locations, true_cone_locations, num_gradient_updates)
        # Plotting the LI kernel (True and Predicted)
        self.plot_LI_kernel_figure(pred_LI_kernel, true_LI_kernel, num_gradient_updates)
        # Plotting the cone mosaic (True and Predicted)
        self.plot_cone_mosaic_figure(pred_cone_mosaic, true_cone_mosaic_numpy, num_gradient_updates)
        # Plotting the ons1, ons2, ons2_pred
        self.plot_ons_figure(ons1, ons2, ons2_pred, num_gradient_updates)


        if not simulating_tetra:
            # log the NS_IP loss
            self.experiment.log_metric('NS_IP Loss', ns_ip_loss.cpu().detach().numpy(), step=num_gradient_updates)

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

                # Unwarp the image to the full field

                internal_percept_sRGB, invalid_regions = cortex.get_unwarped_percept(warped_ip1_sRGB)

                best_coords = largest_valid_region_square(invalid_regions)
                pred_internal_percept_sRGB = internal_percept_sRGB[:, best_coords[0]:best_coords[2], best_coords[1]:best_coords[3]]
            
            self.plot_pred_internal_percept(pred_internal_percept_sRGB, num_gradient_updates)


    def plot_cone_locations_figure(self, pred_cone_locations, true_cone_locations, num_gradient_updates):
        pred_cone_locations_subsampled = pred_cone_locations[:,::4,::4]
        true_cone_locations_subsampled = true_cone_locations[:,::4,::4]

        pred_cone_locations_subsampled[0] -= np.min(pred_cone_locations_subsampled[0])
        pred_cone_locations_subsampled[0] /= np.max(pred_cone_locations_subsampled[0])
        pred_cone_locations_subsampled[1] -= np.min(pred_cone_locations_subsampled[1])
        pred_cone_locations_subsampled[1] /= np.max(pred_cone_locations_subsampled[1])

        true_cone_locations_subsampled[0] -= np.min(true_cone_locations_subsampled[0])
        true_cone_locations_subsampled[0] /= np.max(true_cone_locations_subsampled[0])
        true_cone_locations_subsampled[1] -= np.min(true_cone_locations_subsampled[1])
        true_cone_locations_subsampled[1] /= np.max(true_cone_locations_subsampled[1])

        pred_cone_locations_subsampled = np.transpose(pred_cone_locations_subsampled, (1,2,0)).reshape(-1, 2)
        true_cone_locations_subsampled = np.transpose(true_cone_locations_subsampled, (1,2,0)).reshape(-1, 2)

        fig = plt.figure(figsize=(10,10))
        plt.scatter(pred_cone_locations_subsampled[:,0], pred_cone_locations_subsampled[:,1], c='orange', s=5, zorder=2)
        plt.scatter(true_cone_locations_subsampled[:,0], true_cone_locations_subsampled[:,1], c='green', s=10, zorder=1)
        plt.axis('equal')
        self.experiment.log_figure('00: Cone Locations', fig, step=num_gradient_updates)
        plt.close()
    

    def plot_eye_movement_figure(self, pred_eye_movement, true_eye_movement, num_gradient_updates):

        fig = plt.figure(figsize=(5,5))
        xs = np.stack([pred_eye_movement[:,0,0], true_eye_movement[:,0,0]], 0).reshape(2,-1)
        ys = np.stack([pred_eye_movement[:,0,1], true_eye_movement[:,0,1]], 0).reshape(2,-1)
        plt.plot(xs, ys, color='black')
        plt.scatter(true_eye_movement[:,0,0], true_eye_movement[:,0,1], c='green', s=120)
        plt.scatter(pred_eye_movement[:,0,0], pred_eye_movement[:,0,1], c='orange', s=60)
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        self.experiment.log_figure('01: Motion Error', fig, step=num_gradient_updates)
        plt.close()

        fig = plt.figure(figsize=(5,5))
        diff = (true_eye_movement - pred_eye_movement)
        plt.scatter(diff[:,0,0], diff[:,0,1], s=120)
        plt.xlim([-20,20])
        plt.ylim([-20,20])
        self.experiment.log_figure('02: Motion Diff', fig, step=num_gradient_updates)
        plt.close()

        
    def plot_LI_kernel_figure(self, pred_LI_kernel, true_LI_kernel, num_gradient_updates):

        pred_min_val = np.min(pred_LI_kernel)
        pred_max_val = np.max(pred_LI_kernel)
        pred_range = np.max([np.abs(pred_min_val), pred_max_val])

        true_min_val = np.min(true_LI_kernel)
        true_max_val = np.max(true_LI_kernel)
        true_range = np.max([np.abs(true_min_val), true_max_val])

        fig = plt.figure(figsize=(5,5))
        plt.imshow(pred_LI_kernel, cmap='bwr', vmin=-pred_range, vmax=pred_range)
        plt.colorbar()
        self.experiment.log_figure('03: LI kernel (Pred)', fig, step=num_gradient_updates)
        plt.close()

        fig = plt.figure(figsize=(5,5))
        plt.imshow(true_LI_kernel, cmap='bwr', vmin=-true_range, vmax=true_range)
        plt.colorbar()
        self.experiment.log_figure('03: LI kernel (True)', fig, step=num_gradient_updates)
        plt.close()


    def plot_cone_mosaic_figure(self, pred_cone_mosaic, true_cone_mosaic_numpy, num_gradient_updates):

        fig = plt.figure(figsize=(5,5))
        plt.imshow(pred_cone_mosaic[0].transpose(1,2,0)[self.P:-self.P,self.P:-self.P,:3])
        plt.axis('off')
        self.experiment.log_figure('04: Cone Mosaic (Pred)', fig, step=num_gradient_updates)
        plt.close()

        fig = plt.figure(figsize=(5,5))
        plt.imshow(true_cone_mosaic_numpy[0].transpose(1,2,0)[self.P:-self.P,self.P:-self.P,:3])
        plt.axis('off')
        self.experiment.log_figure('04: Cone Mosaic (True)', fig, step=num_gradient_updates)
        plt.close()


    def plot_ons_figure(self, ons1, ons2, ons2_pred, num_gradient_updates):

        ons1 = ons1[:,:,self.P:-self.P,self.P:-self.P]
        ons2 = ons2[:,:,self.P:-self.P,self.P:-self.P]
        ons2_pred = ons2_pred[:,:,self.P:-self.P,self.P:-self.P]

        min_val = np.min([np.min(ons1), np.min(ons2), np.min(ons2_pred)])
        max_val = np.max([np.max(ons1), np.max(ons2), np.max(ons2_pred)])

        for i in range(ons1.shape[0]):
            fig = plt.figure(figsize=(5,5))
            plt.imshow(ons1[i,0], cmap='gray', vmin=min_val, vmax=max_val)
            plt.colorbar()
            self.experiment.log_figure(f'05: {i:02d} ONS1 (True)', fig, step=num_gradient_updates)
            plt.close()

            fig = plt.figure(figsize=(5,5))
            plt.imshow(ons2_pred[i,0], cmap='gray', vmin=min_val, vmax=max_val)
            plt.colorbar()
            self.experiment.log_figure(f'05: {i:02d} ONS2 (Pred)', fig, step=num_gradient_updates)
            plt.close()

            fig = plt.figure(figsize=(5,5))
            plt.imshow(ons2[i,0], cmap='gray', vmin=min_val, vmax=max_val)
            plt.colorbar()
            self.experiment.log_figure(f'05: {i:02d} ONS2 (True)', fig, step=num_gradient_updates)
            plt.close()


    def plot_pred_internal_percept(self, pred_internal_percept_sRGB, num_gradient_updates):

        for i in range(pred_internal_percept_sRGB.shape[0]):
            fig = plt.figure(figsize=(5,5))
            pred_ip = np.clip(pred_internal_percept_sRGB[i], 0, 1)
            plt.imshow(pred_ip)
            self.experiment.log_figure(f'06: {i:02d} Internal Percept (Pred)', fig, step=num_gradient_updates)
            plt.close()