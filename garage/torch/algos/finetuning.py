import numpy as np
import torch


from garage.torch.algos import MTSAC, PPO
from garage.torch import as_torch_dict

import wandb

class Finetuning_SAC(MTSAC):

    def __init__(self, **sac_kwargs):
        super().__init__(**sac_kwargs)
        
    
class Finetuning_PPO(PPO):

    def __init__(self, **ppo_kwargs):
        super().__init__(**ppo_kwargs)
