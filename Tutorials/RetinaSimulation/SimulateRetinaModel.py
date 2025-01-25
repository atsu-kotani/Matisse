import sys
import os

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Tutorials.RetinaSimulation.helper import *
from Simulated.Retina.RetinaModel import RetinaModel


def run_retina_sim(image_path):

    print (f'Instantiating RetinaModel...')

    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    # Load RetinaModel from the config yaml file
    with open(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Config/Default_Retina_Simulation.yaml', 'r') as f:
        params = yaml.safe_load(f)
    retina = RetinaModel(params, DEVICE)

    print (f'Formating the input image...')

    # You can change the example_image_path to the path of your own image
    example_sRGB_image = load_sRGB_image(retina, image_path, params)
    example_linsRGB_image = retina.CST.sRGB_to_linsRGB(example_sRGB_image)
    example_LMS_image = retina.CST.linsRGB_to_LMS(example_linsRGB_image)
    example_LMS_image = example_LMS_image.unsqueeze(0).permute(0, 3, 1, 2)

    example_LMS_image = example_LMS_image.to(DEVICE)

    print (f'Running Retina Simulation...')

    with torch.no_grad():
        [optic_nerve_signals, _, spatially_sampled_LMS, photoreceptor_activation, bipolar_signals, LMS_current_FoV] = retina.forward(example_LMS_image, intermediate_outputs=True)
        linsRGB_current_FoV = retina.CST.LMS_to_linsRGB(LMS_current_FoV.permute(0,1,3,4,2))
        sRGB_current_FoV = retina.CST.linsRGB_to_sRGB(linsRGB_current_FoV).permute(0,1,4,2,3)

    print (f'Visualizing the simulated retinal responses...')

    # Visualize the simulated retinal responses
    os.makedirs(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results', exist_ok=True)
    render_retinal_signals(optic_nerve_signals, 'Optic Nerve Signals', 'ons') # -> save the gif file as Results/ons.
    render_retinal_signals(bipolar_signals, 'Bipolar Signals', 'bipolar') # -> save the gif file as Results/bipolar.gif
    render_retinal_signals(photoreceptor_activation, 'Photoreceptor Activation', 'pa') # -> save the gif file as Results/pa.gif
    render_retinal_signals(spatially_sampled_LMS, 'Spatially Sampled LMS', 'warped_LMS') # -> save the gif file as Results/warped_LMS.gif
    render_retinal_signals(sRGB_current_FoV, 'Current FoV (sRGB)', 'sRGB_current_FoV') # -> save the gif file as Results/sRGB_current_FoV.gif
    render_retinal_signals(LMS_current_FoV, 'Current FoV (LMS)', 'LMS_current_FoV') # -> save the gif file as Results/LMS_current_FoV.gif


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--image_path', type=str, default=f'{ROOT_DIR}/Tutorials/data/sample_sRGB_image.png')
    args = argument_parser.parse_args()
    run_retina_sim(args.image_path)
