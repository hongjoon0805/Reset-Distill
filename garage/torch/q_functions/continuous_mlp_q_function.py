"""This modules creates a continuous Q-function network."""

import torch
from torch import nn

from garage.torch.modules import MLPModule


class ContinuousMLPQFunction(MLPModule):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, infer = False, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """

        self._multi_input = False
        self._env_spec = env_spec
        if isinstance(env_spec, list):
            self._multi_input = True
            self._obs_dim = 0
            self._action_dim = 0
            for spec in env_spec:
                self._obs_dim += spec.observation_space.flat_dim
                self._action_dim += spec.action_space.flat_dim
            
            self._total_dim = self._obs_dim + self._action_dim
            
            self._zero_pad_per_task = []
            prev_dim = 0
            for spec in env_spec:
                obs_dim = spec.observation_space.flat_dim
                action_dim = spec.action_space.flat_dim
                total_dim = obs_dim + action_dim
                
                self._zero_pad_per_task.append(nn.ConstantPad1d((prev_dim, self._total_dim- (prev_dim + total_dim)),0))
                prev_dim += total_dim



        else:
            self._obs_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=1,
                           **kwargs)
        
        self._feature = None
        feature_size = kwargs['hidden_sizes'][-1]
        if infer:
            self._infer = nn.Linear(feature_size, feature_size)
        

    # pylint: disable=arguments-differ
    def forward(self, observations, actions, seq_idx = None):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        input = torch.cat([observations, actions], 1)
        if self._multi_input:
            
            if isinstance(seq_idx, int):
                zero_pad = self._zero_pad_per_task[seq_idx]
                input = zero_pad(input)
            else:
                batch_size = len(observations)
                new_input = []
                for i in range(batch_size):
                    idx = seq_idx[i]
                    zero_padding = self._zero_pad_per_task[idx]
                    obs = observations[i]
                    new_obs = zero_padding(obs)
                    new_input.append(new_obs)
                input = torch.stack(new_input)
                
        ret = super().forward(input)
        if isinstance(ret, list):
            ret = ret[0]
        return ret

    # InFeR: must be called after forward()
    def get_feature_prediction(self):
        return self._infer(self._feature)
    
    def update_kb(self, policy_kb):

        kb_state_dict = policy_kb.state_dict().clone()
        self.policy_kb.load_state_dict(kb_state_dict)

        print('(garage/torch/q_functions/continuous_mlp_q_functino.py) Knowledge Base Updated!')

