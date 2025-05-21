"""GaussianMLPPolicy."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class GaussianMLPPolicy(StochasticPolicy):
    """MLP whose outputs are fed into a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

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
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 n_tasks,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 name='GaussianMLPPolicy',
                 infer=False, 
                 adaptor=False, 
                 zero_alpha=False,
                 ReDo=False,
                 no_stats=False):
        super().__init__(env_spec, n_tasks, name='GaussianMLPPolicy')

        self._multi_input = False
        if isinstance(env_spec, list):
            self._multi_input = True
            self._obs_dim = 0
            self._action_dim = []
            for spec in env_spec:
                self._obs_dim += spec.observation_space.flat_dim
                print(spec.observation_space.flat_dim)
                self._action_dim.append(spec.action_space.flat_dim)
            
            
            self._zero_pad_per_task = []
            prev_dim = 0
            for spec in env_spec:
                obs_dim = spec.observation_space.flat_dim
                
                self._zero_pad_per_task.append(nn.ConstantPad1d((prev_dim, self._obs_dim - (prev_dim + obs_dim)),0))
                prev_dim += obs_dim


        else:
            self._obs_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim


        self._module = GaussianMLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            n_tasks=n_tasks,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization, 
            adaptor=adaptor, 
            zero_alpha=zero_alpha,
            ReDo=ReDo,
            no_stats=no_stats)
        
        self._feature = None
        if infer:
            feature_size = hidden_sizes[-1]
            self._infer = nn.Linear(feature_size, feature_size)

    def forward(self, observations, seq_idx, features=None):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        if self._multi_input:
            if isinstance(seq_idx, int):
                zero_pad = self._zero_pad_per_task[seq_idx]
                observations = zero_pad(observations)
            else:
                batch_size = len(observations)
                new_observations = []
                for i in range(batch_size):
                    idx = seq_idx[i]
                    zero_padding = self._zero_pad_per_task[idx]
                    obs = observations[i]
                    new_obs = zero_padding(obs)
                    new_observations.append(new_obs)
                observations = torch.stack(new_observations)
        dist = self._module(observations, seq_idx=seq_idx, features=features)
        self._feature = self._module._feature
        self._features = self._module._features
        self._stats = self._module._stats
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))
    
    # InFeR: must be called after forward()
    def get_feature_prediction(self):
        return self._infer(self._feature)
    def reset_parameter(self, column=True, adaptor=False):
        self._module.reset_parameter(column=column, adaptor=adaptor)
