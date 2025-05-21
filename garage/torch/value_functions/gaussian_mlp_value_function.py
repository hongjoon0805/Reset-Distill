"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.value_function import ValueFunction


class GaussianMLPValueFunction(ValueFunction):
    """Gaussian MLP Value Function with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 name='GaussianMLPValueFunction',
                 infer=False,):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        self._multi_input = False
        self._env_spec = env_spec
        if isinstance(env_spec, list):
            self._multi_input = True
            self._obs_dim = 0
            for spec in env_spec:
                self._obs_dim += spec.observation_space.flat_dim
            
            
            self._zero_pad_per_task = []
            prev_dim = 0
            for spec in env_spec:
                obs_dim = spec.observation_space.flat_dim
                
                self._zero_pad_per_task.append(nn.ConstantPad1d((prev_dim, self._obs_dim- (prev_dim + obs_dim)),0))
                prev_dim += obs_dim
            

        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        
        input_dim = self._obs_dim
        output_dim = 1

        self.module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            n_tasks=1,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)
        
        self._feature = None
        if infer:
            feature_size = hidden_sizes[-1]
            self._infer = nn.Linear(feature_size, feature_size)

    def compute_loss(self, obs, returns, seq_idx=None):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        if self._multi_input:
            if isinstance(seq_idx, int):
                zero_pad = self._zero_pad_per_task[seq_idx]
                obs = zero_pad(obs)
            else:
                batch_size = len(obs)
                new_observations = []
                for i in range(batch_size):
                    idx = seq_idx[i]
                    zero_padding = self._zero_pad_per_task[idx]
                    observation = obs[i]
                    new_obs = zero_padding(observation)
                    new_observations.append(new_obs)
                obs = torch.stack(new_observations)
        dist = self.module(obs, seq_idx=0)
        self._feature = self.module._feature
        ll = dist.log_prob(returns.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs, seq_idx = None):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        if self._multi_input:
            zero_pad = self._zero_pad_per_task[seq_idx]
            obs = zero_pad(obs)
        x = self.module(obs, seq_idx=0).mean.flatten(-2)
        self._feature = self.module._feature
        self._features = self.module._features
        self._stats = self.module._stats
        return x
    
    # InFeR: must be called after forward()
    def get_feature_prediction(self):
        return self._infer(self._feature)
    
