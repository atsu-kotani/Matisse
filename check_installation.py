import yaml
import torch
from root_config import ROOT_DIR
from Simulated.Retina.RetinaModel import RetinaModel
from Simulated.Cortex.CortexModel import CortexModel


def check_installation(params):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print ("Checking to instantiate the Retina model...")
    retina          = RetinaModel(params, device).to(device)
    print ("Retina model instantiated successfully!")
    print ("Checking to instantiate the Cortex model...")
    cortex          = CortexModel(params, device).to(device)
    print ("Cortex model instantiated successfully!")
    print ("Checking to compile the Cortex model...")
    cortex          = torch.compile(cortex)
    print ("Cortex model compiled successfully!")
    print ("**************************************************")
    print("             Installation successful!")
    print ("**************************************************")

if __name__ == '__main__':
    with open(f'{ROOT_DIR}/Experiment/Config/Default/LMS.yaml', 'r') as f:
        params = yaml.safe_load(f)
    check_installation(params)
