# Load root configuration
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', '..'))
from root_config import ROOT_DIR

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
import imageio
import IPython
import numpy as np
from PIL import Image
import subprocess
import argparse

# Pre-define the helper functions for later use

def load_sRGB_image(retina, image_path, params):
    # Loading the input stimulus
    MSS = params['RetinaModel']['max_shift_size']
    current_image_resolution = retina.required_image_resolution + MSS * 6

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
    image_tensor = torch.FloatTensor(image).to(retina.device)
    return image_tensor


def render_retinal_signals(signals, image_name='Optic Nerve Signals', filename=None):
    # Visualizing the retina simulation
    images = []
    val_min = signals.min()
    val_max = signals.max()
    
    print (f'Rendering {image_name}...')

    # skip the first 5 frames
    signals = signals[:, 5:]
    for i in range(signals.shape[1]):
        fig, ax = plt.subplots(figsize=(5,5))

        if len(signals[0, i].shape) == 3:
            image = signals[0, i].detach().permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1)
            # removing the bordering pixels to remove artifacts from simulating the limited FoV
            ax.imshow((image[1:-1,1:-1,:3] * 255).astype(np.uint8))
        else:
            # removing the bordering pixels to remove artifacts from simulating the limited FoV
            ax.imshow(signals[0, i, 1:-1, 1:-1].detach().cpu().numpy(), cmap='gray', vmin=val_min, vmax=val_max)

        # remove the white space around the image
        ax.set_axis_off()
        # completely remove the white space around the image
        fig.tight_layout()
        # ax.set_title(f'{image_name}')
        fig.canvas.draw()

        os.makedirs(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/retina_sim/{filename}', exist_ok=True)
        plt.savefig(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/retina_sim/{filename}/{i:04d}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Save images as a gif with looping
    # make gif from images
    images = []
    for i in range(signals.shape[1]):
        images.append(imageio.imread(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/retina_sim/{filename}/{i:04d}.png'))
    imageio.mimsave(f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/{filename}.gif', images, fps=20)

    # render a mp4 video from images using ffmpeg
    subprocess.run(['ffmpeg', '-y', '-framerate', '20', '-i', f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/retina_sim/{filename}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{ROOT_DIR}/Tutorials/RetinaSimulation/Results/retina_sim/{filename}.mp4'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
