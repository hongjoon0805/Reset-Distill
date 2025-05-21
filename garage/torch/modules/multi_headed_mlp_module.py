"""MultiHeadedMLPModule."""
import copy

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math

from garage.torch import NonLinearity, CReLU
from garage.torch.modules.noisy_net import NoisyLinear

class MultiHeadedMLPModule(nn.Module):
    """MultiHeadedMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 n_tasks,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False, 
                 noisy_net=False, 
                 adaptor=False, 
                 zero_alpha=False, 
                 ReDo=False,
                 no_stats=False):
        super().__init__()

        self._layers = nn.ModuleList()
        self._use_adaptor = adaptor

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads, n_tasks)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads, n_tasks)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads, n_tasks)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads, n_tasks)

        self._layers = nn.ModuleList()
        if self._use_adaptor:
            self._adaptors = nn.ModuleList()
            self._alphas = []

        self._noisy_layers = []
        self._hidden_nonlinearity = NonLinearity(hidden_nonlinearity)
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._all_hidden_sizes = 0

        self._stats = {}
        self._stats['zero_ratio'] = []
        self._stats['dormant'] = []
        self._stats['zero_idx'] = {}
        self._stats['dormant_idx'] = {}
        self._ReDo = ReDo
        self._no_stats = no_stats

        prev_size = input_dim
        # CReLU
        if hidden_nonlinearity is CReLU:
            for hidden_size in hidden_sizes:
                hidden_layers = nn.Sequential()
                size = hidden_size // 2
                if layer_normalization:
                    hidden_layers.add_module('layer_normalization',
                                            nn.LayerNorm(prev_size))
                if noisy_net:
                    linear_layer = NoisyLinear(prev_size, size)
                    linear_layer.reset_parameters()
                    self._noisy_layers.append(linear_layer)
                else:
                    linear_layer = nn.Linear(prev_size, size)
                    hidden_w_init(linear_layer.weight)
                    hidden_b_init(linear_layer.bias)

                hidden_layers.add_module('linear', linear_layer)

                hidden_layers.add_module('non_linearity',
                                        NonLinearity(hidden_nonlinearity))

                self._layers.append(hidden_layers)
                prev_size = hidden_size
            
        # ReLU
        else:
            for idx, size in enumerate(hidden_sizes):
                hidden_layers = nn.Sequential()
                self._all_hidden_sizes += size
                if layer_normalization:
                    hidden_layers.add_module('layer_normalization',
                                            nn.LayerNorm(prev_size))
                
                if noisy_net:
                    linear_layer = NoisyLinear(prev_size, size)
                    linear_layer.reset_parameters()
                    self._noisy_layers.append(linear_layer)
                else:
                    linear_layer = nn.Linear(prev_size, size)
                    hidden_w_init(linear_layer.weight)
                    hidden_b_init(linear_layer.bias)

                    if self._use_adaptor:

                        adaptor_layer = Adaptor(prev_size, size, NonLinearity(hidden_nonlinearity), zero_alpha=zero_alpha)
                        adaptor_layer.reset_parameter(hidden_w_init, hidden_b_init)


                hidden_layers.add_module('linear', linear_layer)

                if hidden_nonlinearity and not self._use_adaptor:
                    hidden_layers.add_module('non_linearity',
                                            NonLinearity(hidden_nonlinearity))                    

                
                self._layers.append(hidden_layers)
                
                if self._use_adaptor:
                    self._adaptors.append(adaptor_layer)
                prev_size = size

        self._output_layers = nn.ModuleList()
        
        for i in range(n_heads*n_tasks):
            output_layer = nn.Sequential()
            if noisy_net:
                linear_layer = NoisyLinear(prev_size, output_dims[i])
                linear_layer.reset_parameters()
                self._noisy_layers.append(linear_layer)
            else:
                linear_layer = nn.Linear(prev_size, output_dims[i])
                output_w_inits[i](linear_layer.weight)
                output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module('non_linearity',
                                        NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads, n_tasks):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * (n_heads * n_tasks)
            if (len(var) == n_heads) and (n_heads != n_tasks):
                return var * n_tasks
            if len(var) == n_tasks:
                ret = []
                for t in range(n_tasks):
                    for _ in range(n_heads):
                        ret.append(var[t])
                return ret


            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads*n_tasks, var))
        return [copy.deepcopy(var) for _ in range(n_heads*n_tasks)]

    # pylint: disable=arguments-differ
    def forward(self, input_val, features = None):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        self._features = None
        use_adaptor = (features is not None) and (self._use_adaptor)
        if use_adaptor:

            zero_cnt = 0
            dormant_cnt = 0
            for idx, (layer, adaptor) in enumerate(zip(self._layers, self._adaptors)):
                h = 0
                if idx > 0:
                    feature = features[idx-1]
                    h = adaptor(feature)
                
                x = layer(x)
                x = self._hidden_nonlinearity(x + h)

                if (self._no_stats == False) or self._ReDo:

                    tau = -0.9950

                    x_detached = x.detach()

                    if len(x_detached.shape) == 3:
                        x_averaged = x_detached.mean(dim=(0,1))
                    else:
                        x_averaged = x_detached.mean(dim=0)

                    x_zero_cnt = int((x_averaged == 0).sum())
                    zero_cnt += x_zero_cnt

                    dormant = int((x_averaged <= tau).sum())
                    dormant_cnt += dormant

                    self._stats['zero_idx'][idx] = (x_averaged == 0)
                    self._stats['dormant_idx'][idx] = (x_averaged <= tau)
        
        else:
            self._features = []
            zero_cnt = 0
            dormant_cnt = 0
            
            for idx, layer in enumerate(self._layers):
                x = layer(x)
                if self._use_adaptor:
                    x = self._hidden_nonlinearity(x)

                if (self._no_stats == False) or self._ReDo:
                    tau = -0.9950

                    x_detached = x.detach()

                    if len(x_detached.shape) == 3:
                        x_averaged = x_detached.mean(dim=(0,1))
                    else:
                        x_averaged = x_detached.mean(dim=0)

                    x_zero_cnt = int((x_averaged == 0).sum())
                    zero_cnt += x_zero_cnt

                    dormant = int((x_averaged <= tau).sum())
                    dormant_cnt += dormant

                    self._stats['zero_idx'][idx] = (x_averaged == 0)
                    self._stats['dormant_idx'][idx] = (x_averaged <= tau)

                self._features.append(x)
        
        self._feature = x
        if self._no_stats == False or self._ReDo:
            self._stats['zero_ratio'].append(zero_cnt/self._all_hidden_sizes)
            self._stats['dormant'].append(dormant_cnt/self._all_hidden_sizes)

        return [output_layer(x) for output_layer in self._output_layers]

    def reset_noise(self):
        for layers in self._noisy_layers:
            layers.reset_noise()
    def reset_parameter(self, column=True, adaptor=False):
        if column:
            for layer in self._layers:
                for p in layer.parameters():
                    if len(p.shape) > 1:
                        self._hidden_w_init(p)
                        print('(garage/torch/modules/multi_headed_mlp_module.py) Reset column weight')
                    else:
                        self._hidden_b_init(p)
                        print('(garage/torch/modules/multi_headed_mlp_module.py) Reset column bias')

        if adaptor:
            for ad in self._adaptors:
                ad.reset_parameter(self._hidden_w_init, self._hidden_b_init)
                print('(garage/torch/modules/multi_headed_mlp_module.py) Reset adaptor')


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out    

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        # self.mu = mu.cuda()
        # self.rho = rho.cuda()

        self.mu = mu
        self.rho = rho

        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        # epsilon = self.normal.sample(self.mu.size()).cuda()
        epsilon = self.normal.sample(self.mu.size())
        return self.mu + self.sigma * epsilon   

RATIO = 0.25

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, ratio=RATIO):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight_mu)
        gain = 1 # Var[w] + sigma^2 = 2/fan_inㄴ
        
        total_var = 2 / fan_in
        noise_var = total_var * ratio
        mu_var = total_var - noise_var
        
        noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
        bound = math.sqrt(3.0) * mu_std
        rho_init = np.log(np.exp(noise_std)-1)
        
        nn.init.uniform_(self.weight_mu, -bound, bound)
        self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0))
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)

    def forward(self, input, bayes_sample=False):
        if bayes_sample:
            weight = self.weight.sample()
            bias = self.bias
        else:
            weight = self.weight.mu
            bias = self.bias

        return F.linear(input, weight, bias)
    

class Adaptor(nn.Module):
    def __init__(self, in_features, out_features, non_linearity, zero_alpha=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_V = nn.Parameter(torch.Tensor(in_features, in_features))
        self.weight_U = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_c = nn.Parameter(torch.Tensor(in_features))

        self._zero_alpha = zero_alpha

        if zero_alpha:
            self.alpha=0
            print('(garage/torch/modules/multi_headed_mlp_module.py) Alpha is zero!!')
        else:
            self.alpha = nn.Parameter(torch.Tensor(out_features))

        self.non_linearity = non_linearity
        if non_linearity is None:
            self.non_linearity = nn.ReLU()

    def forward(self, input):
        h = F.linear(input, self.weight_V, self.bias_c)
        h = self.non_linearity(h)
        h = self.alpha * F.linear(h, self.weight_U)

        return h

    def reset_parameter(self, hidden_w_init, hidden_b_init):

        hidden_w_init(self.weight_V)
        hidden_w_init(self.weight_U)
        hidden_b_init(self.bias_c)
        if not self._zero_alpha:
            nn.init.uniform_(self.alpha, a=0.0, b=0.1)




        