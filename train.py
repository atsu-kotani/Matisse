import comet_ml
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import yaml
import torch
import shutil
import argparse
from tqdm.auto import tqdm
import torch.optim as optim
from Experiment.helper import *
from Dataset import create_dataset
from root_config import PIN_MEMORY, ROOT_DIR
from Simulated.Retina.RetinaModel import RetinaModel
from Simulated.Cortex.CortexModel import CortexModel


def train_cortical_model(params):

    experiment_name = params['Experiment']['name']

    os.makedirs(f'{ROOT_DIR}/Experiment/LearnedWeights/{experiment_name}', exist_ok=True)
    # copy the config file to the experiment folder
    shutil.copy(f'{ROOT_DIR}/Experiment/Config/{args.config_filename}.yaml', f'{ROOT_DIR}/Experiment/LearnedWeights/{experiment_name}/config.yaml')

    print ("*"*50)
    print (f'Experiment Name = {experiment_name}')
    print ("*"*50)


    # Instantiate the retina simulation model
    retina          = RetinaModel(params, DEVICE)

    # Instantiate the cortex simulation model
    cortex          = CortexModel(params, DEVICE)
    cortex          = torch.compile(cortex)

    # Prepare the dataset
    dataset         = create_dataset(params['Dataset']['dataset_name'], params, retina)
    trainloader     = torch.utils.data.DataLoader(dataset, batch_size=params['Dataset']['batch_size'], shuffle=True, num_workers=8, pin_memory=PIN_MEMORY, persistent_workers=True)

    # load cortex model if dimensionality boosting is enabled
    if params['DimensionalityBoosting']['is_dimensionality_boosting']: # Dimensionality boosting
        previous_model_name = get_previous_model_name(params)
        cortex.load_state_dict(torch.load(f'{ROOT_DIR}/Experiment/LearnedWeights/{previous_model_name}/{params["DimensionalityBoosting"]["load_pretrain_timestep"]}.pt', weights_only=True))
        
    # training setting
    # Learnable parameters in the cortex model (updated during the main training loop)
    params_list = [{'params': cortex.C_cone_spectral_type.parameters()},
                   {'params': cortex.D_demosaicing.parameters()},
                   {'params': cortex.W_lateral_inhibition_weights.parameters()},
                   {'params': cortex.P_cell_position.parameters()}]
    main_optimizer  = optim.Adam(params_list, lr=params['Training']['learning_rate'])
    main_optimizer.zero_grad(set_to_none=True)

    # Neural Scope for cone mosaic
    ns_cm_optimizer = optim.Adam(cortex.ns_cm.parameters(), lr=params['Training']['learning_rate'])
    ns_cm_optimizer.zero_grad(set_to_none=True)

    # Neural Scope for internal percept (IP)
    ns_ip_optimizer = optim.Adam(cortex.ns_ip.parameters(), lr=params['Training']['learning_rate'])
    ns_ip_optimizer.zero_grad(set_to_none=True)

    logging_timesteps = []
    for i in range(0, 2000, 100):
        logging_timesteps.append(i)
    for i in range(2000, 10000, 1000):
        logging_timesteps.append(i)
    for i in range(10000, params['Training']['max_gradient_updates']+1, 10000):
        logging_timesteps.append(i)


    if params['Training']['learning_progress_logging']:
        from Experiment.ProgressLogger import create_logger
        logger = create_logger(params['Training']['logging_mode'], experiment_name, retina.required_image_resolution)

        # prepare ground truth retinal parameters
        true_LI_kernel = retina.LateralInhibition.get_LI_kernel().cpu().detach().numpy()
        true_cone_locations = retina.SpatialSampling.cone_locs.clone().permute(2,0,1).cpu().detach().numpy()

    # Main training preparation
    epoch = 0
    num_gradient_updates = 0
    simulating_tetra = params['Experiment']['simulating_tetra']

    # if your training accidently got interrupted, you can resume from the last saved timestep
    cortex, main_optimizer, ns_cm_optimizer, ns_ip_optimizer, num_gradient_updates = load_existing_timestep(params, experiment_name, cortex, main_optimizer, ns_cm_optimizer, ns_ip_optimizer, logging_timesteps)

    # Get the ground truth retinal parameters
    true_LI_kernel_size     = retina.LateralInhibition.get_kernel_size()
    true_cone_mosaic        = retina.SpectralSampling.get_cone_mosaic()
    true_cone_mosaic_numpy  = true_cone_mosaic.cpu().detach().numpy()

    bar = tqdm(total=params['Training']['max_gradient_updates'])
    bar.set_description(f"Gradient Updates:")

    # Main training loop
    while True:
        epoch += 1

        for batch_LMS_full in trainloader:

            batch_LMS_full = batch_LMS_full.permute(0,3,1,2).to(DEVICE) # (BS, C, RIR+2*MSS, RIR+2*MSS)

            # Run the retina simulation, independently from the cortical learning
            with torch.no_grad():
                batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV = retina.forward(batch_LMS_full)
                batch_ons1, batch_ons2 = batch_ons[:,0:1], batch_ons[:,1:2]
                batch_warped_linsRGB1 = retina.CST.LMS_to_linsRGB(batch_warped_LMS_current_FoV[:,0].permute(0,2,3,1)).permute(0,3,1,2)

            main_loss, ns_cm_loss, ns_ip_loss, pred_cone_mosaic, ons2_pred, pred_dxy = cortex.main_train(batch_ons1, batch_ons2, batch_warped_linsRGB1, batch_true_dxy, true_cone_mosaic, true_LI_kernel_size)

            # Backward pass
            main_loss.backward()
            main_optimizer.step()
            main_optimizer.zero_grad(set_to_none=True)

            ns_cm_loss.backward()
            ns_cm_optimizer.step()
            ns_cm_optimizer.zero_grad(set_to_none=True)

            if not simulating_tetra:
                ns_ip_loss.backward()
                ns_ip_optimizer.step()
                ns_ip_optimizer.zero_grad(set_to_none=True)

            if num_gradient_updates in logging_timesteps:
                torch.save(cortex.state_dict(), f'{ROOT_DIR}/Experiment/LearnedWeights/{experiment_name}/{num_gradient_updates}.pt')

            # Logging the learning progress if enabled
            if params['Training']['learning_progress_logging'] and num_gradient_updates in logging_timesteps:
                if params['Training']['logging_mode'] == 'Comet':
                    true_eye_movement   = batch_true_dxy.cpu().detach().numpy() * (retina.required_image_resolution / 2)
                    pred_eye_movement   = pred_dxy.cpu().detach().numpy() * (retina.required_image_resolution / 2)

                    pred_LI_kernel      = cortex.W_lateral_inhibition_weights.get_predicted_kernel(true_LI_kernel_size).cpu().detach().numpy()
                    pred_cone_locations = cortex.P_cell_position.get_XY_default_locations()[0].cpu().detach().numpy()
                    
                    pred_cone_mosaic    = pred_cone_mosaic.cpu().detach().numpy()
                    pred_cone_mosaic    = np.clip(pred_cone_mosaic, 0, 1)

                    ons1_numpy          = batch_ons1.cpu().detach().numpy()
                    ons2_numpy          = batch_ons2.cpu().detach().numpy()
                    ons2_pred_numpy     = ons2_pred.cpu().detach().numpy()

                    main_loss_numpy     = main_loss.cpu().detach().numpy()
                    ns_cm_loss_numpy    = ns_cm_loss.cpu().detach().numpy()

                    logger.log_progress(simulating_tetra, retina, cortex, num_gradient_updates, 
                                        main_loss_numpy, ns_cm_loss_numpy, ns_ip_loss,
                                        true_eye_movement, pred_eye_movement,
                                        true_LI_kernel, pred_LI_kernel,
                                        true_cone_mosaic_numpy, pred_cone_mosaic,
                                        true_cone_locations, pred_cone_locations,
                                        ons1_numpy, ons2_numpy, ons2_pred_numpy)

                elif params['Training']['logging_mode'] == 'Local':
                    logger.log_progress(simulating_tetra, retina, cortex, num_gradient_updates, main_loss, ns_cm_loss, ns_ip_loss, params['RetinaModel']['retina_spatial_sampling'])

            # Update the progress bar
            bar.set_postfix(loss=main_loss.item())
            num_gradient_updates += 1
            bar.update(1)

            # End the training loop if the maximum number of gradient updates is reached
            if num_gradient_updates == params['Training']['max_gradient_updates'] + 1:
                break

        if DEVICE == 'cuda:0':
            torch.cuda.empty_cache()

        if num_gradient_updates == params['Training']['max_gradient_updates'] + 1:
            break

    bar.close()

    # save the final model
    torch.save(cortex.state_dict(), f'{ROOT_DIR}/Experiment/LearnedWeights/{experiment_name}/{params["Training"]["max_gradient_updates"]}.pt')

    logger.generate_progress_video()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YAML configuration file for experiment.')
    parser.add_argument('-f','--config_filename', dest='config_filename', default='Default/LMS')
    args = parser.parse_args()

    with open(f'{ROOT_DIR}/Experiment/Config/{args.config_filename}.yaml', 'r') as f:
        params = yaml.safe_load(f)

    train_cortical_model(params)
