"""PyTorch-backed modules and algorithms."""
# yapf: disable
from garage.torch._functions import (as_torch_dict, compute_advantages,
                                     expand_var, filter_valids, flatten_batch,
                                     flatten_to_single_vector, global_device,
                                     NonLinearity, CReLU, np_to_torch,
                                     output_height_2d, output_width_2d,
                                     pad_to_last, prefer_gpu,
                                     product_of_gaussians, set_gpu_mode,
                                     soft_update_model, state_dict_to,
                                     torch_to_np, update_module_params)

# yapf: enable
__all__ = [
    'NonLinearity',
    'CReLU',
    'as_torch_dict',
    'compute_advantages',
    'expand_var',
    'filter_valids',
    'flatten_batch',
    'flatten_to_single_vector',
    'global_device',
    'np_to_torch',
    'output_height_2d',
    'output_width_2d',
    'pad_to_last',
    'prefer_gpu',
    'product_of_gaussians',
    'set_gpu_mode',
    'soft_update_model',
    'state_dict_to',
    'torch_to_np',
    'update_module_params',
]
