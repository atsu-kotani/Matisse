import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import requests
from io import BytesIO
from PIL import Image

from root_config import ROOT_DIR
from Dataset.Abstract import Dataset
from Dataset import register_class
from torchvision.datasets import DatasetFolder
import glob


def download_file(url, save_path):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading File")
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
    print(f"File downloaded and saved to {save_path}")


@register_class("ImageNet64x64")
class ImageNet64x64(Dataset):
    def __init__(self, params, retina):
        super(ImageNet64x64, self).__init__(params, retina)

        original_data_dir = f"{ROOT_DIR}/Dataset/ImageNet_64x64/data"
        os.makedirs(original_data_dir, exist_ok=True)

        self.required_dataset_image_resolution = retina.required_image_resolution + 2 * params['RetinaModel']['max_shift_size']

        data_dir = f'{ROOT_DIR}/Dataset/ImageNet_{self.required_dataset_image_resolution}x{self.required_dataset_image_resolution}/LMS'
        os.makedirs(f'{data_dir}/data', exist_ok=True)

        DATASET_SIZE = 5000

        if not os.path.exists(f'{data_dir}/data/image_{DATASET_SIZE-1}.pt'):

            print ("Dataset is not ready -- generating the dataset...")
            print ("This step would take about a minute in total.")
            # Download the dataset from the huggingface repository
            # We would only need to download the dataset once
            dataset_url = (
                "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64/"
                "resolve/main/data/validation-00000-of-00001.parquet?download=true"
            )
            local_parquet_path = os.path.join(original_data_dir, "validation-00000-of-00001.parquet")

            # Download the dataset if not already downloaded
            if not os.path.exists(local_parquet_path):
                print("Downloading dataset...")
                download_file(dataset_url, local_parquet_path)

            # Load and process the Parquet file
            if not os.path.exists(f'{ROOT_DIR}/Dataset/ImageNet_64x64/data/image_49999.png'):
                print("Loading and processing the Parquet file...")
                data = pd.read_parquet(local_parquet_path)
                self.extract_images(data, original_data_dir, DATASET_SIZE)

            # batch convert the images to the required resolution and LMS
            print("Batch converting images to LMS...")
            self.batch_convert_images_to_LMS(original_data_dir, data_dir+'/data', retina, DATASET_SIZE)

            print ("Dataset is ready -- starting the main training loop!")

        # Load the images
        self.all_data = DatasetFolder(root=data_dir, loader=self.loader, extensions='.pt')


    def batch_convert_images_to_LMS(self, data_dir, data_dir_output, retina, DATASET_SIZE):

        device = retina.device

        for image_id in tqdm(range(DATASET_SIZE), desc="Batch converting images to LMS"):
            if not os.path.exists(f'{data_dir_output}/image_{image_id}.pt'):
                image_path = f'{data_dir}/image_{image_id}.png'
                image = Image.open(image_path)

                # center crop the image to the required resolution
                H, W = image.size
                if H > self.required_dataset_image_resolution and W > self.required_dataset_image_resolution:
                    image = image.crop((W//2-self.required_dataset_image_resolution//2, H//2-self.required_dataset_image_resolution//2, W//2+self.required_dataset_image_resolution//2, H//2+self.required_dataset_image_resolution//2))
                else:
                    image = image.resize((self.required_dataset_image_resolution, self.required_dataset_image_resolution))

                image = image.convert('RGB')
                image_np = np.array(image) / 255.0
                image_tensor = torch.FloatTensor(image_np).to(device)
                
                linsRGB_image = retina.CST.sRGB_to_linsRGB(image_tensor)
                LMS = retina.CST.linsRGB_to_LMS(linsRGB_image)

                image_path_output = image_path.replace(data_dir, data_dir_output)
                image_path_output = image_path_output.replace('.png', '.pt')

                # save LMS as pt file
                torch.save(LMS, image_path_output)


    def loader(self, path):
        # load LMS as pt file
        LMS = torch.load(path, map_location='cpu', weights_only=True)
        return LMS


    def extract_images(self, data, output_dir, DATASET_SIZE):
        """Extract images from a Parquet file and save them locally."""
        print("Extracting images...")
        for idx, row in tqdm(data.iterrows(), total=DATASET_SIZE, desc="Extracting Images"):
            try:
                # Extract the bytes from the 'bytes' key in the 'image' column
                img_data = row['image']['bytes']
                img = Image.open(BytesIO(img_data))
                img_save_path = os.path.join(output_dir, f"image_{idx}.png")
                img.save(img_save_path)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
            if idx > DATASET_SIZE:
                break
        print(f"Images extracted and saved to {output_dir}")


    def __getitem__(self, index):
        index = index % len(self.all_data)
        data, _ = self.all_data[index]
        return data


    def __len__(self):
        return len(self.all_data)
