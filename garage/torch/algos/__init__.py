"""PyTorch algorithms."""
# isort:skip_file

# SAC needs to be imported before MTSAC
from garage.torch.algos.sac import SAC
from garage.torch.algos.mtsac import MTSAC
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO
from garage.torch.algos.bc import BC_SAC, BC_PPO
from garage.torch.algos.rnd import RND_SAC, RND_PPO
from garage.torch.algos.finetuning import Finetuning_SAC, Finetuning_PPO

from garage.torch.algos.ewc import EWC_SAC, EWC_PPO
from garage.torch.algos.pandc import P_and_C_SAC, P_and_C_PPO

__all__ = [
    'SAC', 'MTSAC', 'PPO', 'VPG',
    'BC_SAC', 'BC_PPO',
    'RND_SAC', 'RND_PPO', 
    'Finetuning_SAC', 'Finetuning_PPO',
    'EWC_SAC', 'EWC_PPO',
    'P_and_C_SAC', 'P_and_C_PPO'
]
