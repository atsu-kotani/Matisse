import os
import torch
import numpy as np

# initialize random seed
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

# root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# device
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps:0'
else:
    DEVICE = 'cpu'

# device specific settings
if DEVICE == 'cuda:0':
    PIN_MEMORY = True
    NUM_WORKERS = 1
    DATASET_SIZE = 50000
    LOG_SYSTEM_METRICS = True
elif DEVICE == 'mps:0':
    PIN_MEMORY = False
    NUM_WORKERS = 1
    DATASET_SIZE = 10000
    LOG_SYSTEM_METRICS = False
else:
    PIN_MEMORY = False
    NUM_WORKERS = 1
    DATASET_SIZE = 10000
    LOG_SYSTEM_METRICS = False
    torch.set_num_threads(os.cpu_count())
