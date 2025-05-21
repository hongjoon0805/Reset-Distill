import metaworld
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import CReLU

from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction

from garage.torch.algos import BC_SAC, BC_PPO
from garage.torch.algos import RND_SAC, RND_PPO
from garage.torch.algos import Finetuning_SAC, Finetuning_PPO


from garage.torch.algos import EWC_SAC, EWC_PPO
from garage.torch.algos import P_and_C_SAC, P_and_C_PPO

    
def get_algo(args, spec, n_tasks, train_envs, test_envs, train_info, env_seq):
    batch_size, epoch_cycles = train_info
    env_type, rl_method = args.env_type, args.rl_method
    if env_type == 'metaworld' or env_type == "dm_control":

        if rl_method == 'sac':
            
            hidden_nonlinearity = CReLU if args.crelu else nn.ReLU

            hidden_sizes = [256, 256] if env_type == "metaworld" else [1024, 1024]

            policy = TanhGaussianMLPPolicy(
                env_spec=spec,
                n_tasks=n_tasks,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                min_std=np.exp(-20.),
                max_std=np.exp(2.),
                infer=args.infer, 
                adaptor=(args.cl_method == 'pandc'),
                zero_alpha = ((args.cl_method == 'pandc') and args.zero_alpha),
                ReDo = args.ReDo,
                no_stats = args.no_stats
            )
            
            hidden_nonlinearity = CReLU if args.crelu else F.relu

            qf1 = ContinuousMLPQFunction(env_spec=spec,
                                         infer=args.infer,
                                         hidden_sizes=hidden_sizes,
                                         hidden_nonlinearity=hidden_nonlinearity)

            qf2 = ContinuousMLPQFunction(env_spec=spec,
                                         infer=args.infer,
                                         hidden_sizes=hidden_sizes,
                                         hidden_nonlinearity=hidden_nonlinearity)
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6),)
            
            if isinstance(spec,list):
                max_episode_length = spec[0].max_episode_length
            else:
                max_episode_length = spec.max_episode_length
            sampler = LocalSampler(agents=policy,
                                    envs=train_envs,
                                    max_episode_length=max_episode_length,
                                    n_workers=1,)
            
            if env_type == "metaworld":
                sac_kwargs = {
                'policy': policy,
                'qf1': qf1,
                'qf2': qf2,
                'sampler': sampler,
                'seed': args.seed,
                'gradient_steps_per_itr': batch_size,
                'eval_env': test_envs,
                'env_spec': spec,
                'steps_per_epoch': epoch_cycles,
                'replay_buffer': replay_buffer,
                'use_exploration': args.use_exploration,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'crelu': args.crelu,
                'num_tasks': 1,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'log_name': args.proc_name,
                'use_wandb': args.wandb,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,
            }
            
            elif env_type == "dm_control":
                sac_kwargs = {
                'policy': policy,
                'qf1': qf1,
                'qf2': qf2,
                'sampler': sampler,
                'seed': args.seed,
                'gradient_steps_per_itr': batch_size // 4,
                'eval_env': test_envs,
                'env_spec': spec,
                'steps_per_epoch': epoch_cycles,
                'replay_buffer': replay_buffer,
                'use_exploration': args.use_exploration,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'crelu': args.crelu,
                'num_tasks': 1,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'log_name': args.proc_name,
                'use_wandb': args.wandb,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,

                'policy_lr': 1e-4,
                'qf_lr': 1e-4,
                'fixed_alpha': 0.01,
                'buffer_batch_size': 1024,
                'multi_input': True,
            }

            if args.cl_method == 'finetuning':
                algo = Finetuning_SAC(**sac_kwargs)

            if args.cl_method == 'bc':
                algo = BC_SAC(**sac_kwargs,
                            cl_reg_coef=args.cl_reg_coef, 
                            expert_buffer_size=args.expert_buffer_size)
            if args.cl_method == 'rnd':
                algo = RND_SAC(**sac_kwargs,
                                    cl_reg_coef=args.cl_reg_coef,
                                    expert_buffer_size=args.expert_buffer_size,
                                    replay_buffer_size=args.replay_buffer_size,
                                    env_seq=env_seq,
                                    nepochs_offline=args.nepochs_offline,
                                    bc_kl=args.bc_kl, distill_kl=args.distill_kl, reset_offline_actor=args.reset_offline_actor)
            
            if args.cl_method == 'ewc':
                algo = EWC_SAC(**sac_kwargs, cl_reg_coef=args.cl_reg_coef)
            if args.cl_method == 'pandc':
                algo = P_and_C_SAC(**sac_kwargs, cl_reg_coef=args.cl_reg_coef, 
                                   compress_step=args.compress_step, 
                                   bc=args.use_pandc_bc, reset_column=args.reset_column, 
                                   reset_adaptor=args.reset_adaptor)
        elif rl_method == 'ppo':

            hidden_nonlinearity = CReLU if args.crelu else torch.tanh

            hidden_sizes = (128, 128) if env_type == 'metaworld' else (1024, 1024)

            policy = GaussianMLPPolicy(
                env_spec=spec,
                n_tasks=n_tasks,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
                infer=args.infer,
                adaptor=(args.cl_method == 'pandc'),
                zero_alpha = ((args.cl_method == 'pandc') and args.zero_alpha),
                ReDo = args.ReDo,
                no_stats = args.no_stats,
            )

            value_function = GaussianMLPValueFunction(env_spec=spec,
                                                    hidden_sizes=hidden_sizes,
                                                    hidden_nonlinearity=hidden_nonlinearity,
                                                    output_nonlinearity=None,
                                                    infer=args.infer,)

            if isinstance(spec,list):
                max_episode_length = spec[0].max_episode_length
            else:
                max_episode_length = spec.max_episode_length
            
            sampler = LocalSampler(agents=policy,
                                    envs=train_envs,
                                    max_episode_length=max_episode_length,
                                    n_workers=1,)

            if env_type == 'metaworld':
                ppo_kwargs = {
                'env_spec': spec,
                'policy': policy,
                'value_function': value_function,
                'eval_env': test_envs,
                'sampler': sampler,
                'seed': args.seed,
                'discount': 0.99,
                'gae_lambda': 0.95,
                'center_adv': True,
                'lr_clip_range': args.lr_clip_range,
                'log_name': args.proc_name,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'use_wandb': args.wandb,
                'crelu': args.crelu,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,
                'multi_input': False,

            }
                
            elif env_type == 'dm_control':
                ppo_kwargs = {
                'env_spec': spec,
                'policy': policy,
                'value_function': value_function,
                'eval_env': test_envs,
                'sampler': sampler,
                'seed': args.seed,
                'discount': 0.99,
                'gae_lambda': 0.95,
                'center_adv': True,
                'lr_clip_range': args.lr_clip_range,
                'log_name': args.proc_name,
                'num_evaluation_episodes': args.num_evaluation_episodes,
                'q_reset': args.q_reset,
                'policy_reset': args.policy_reset,
                'first_task': args.first_task,
                'use_wandb': args.wandb,
                'crelu': args.crelu,
                'infer': args.infer,
                'wasserstein': args.wasserstein,
                'ReDo': args.ReDo,
                'no_stats': args.no_stats,
                'policy_lr':3e-4,
                'value_lr':3e-4,
                'max_optimization_epochs':64,
                'minibatch_size':1024,
                'multi_input': True

            }


            if args.cl_method == 'finetuning':
                algo = Finetuning_PPO(**ppo_kwargs)
            
            if args.cl_method == 'ewc':
                algo = EWC_PPO(**ppo_kwargs, cl_reg_coef=args.cl_reg_coef)
         
            if args.cl_method == 'pandc':
                algo = P_and_C_PPO(**ppo_kwargs, cl_reg_coef=args.cl_reg_coef, 
                                   compress_step=args.compress_step, 
                                   bc=args.use_pandc_bc, 
                                   reset_column=args.reset_column, 
                                   reset_adaptor=args.reset_adaptor)
            if args.cl_method == 'bc':
                algo = BC_PPO(**ppo_kwargs,
                            cl_reg_coef=args.cl_reg_coef, 
                            expert_buffer_size=args.expert_buffer_size)
            if args.cl_method == 'rnd':
                algo = RND_PPO(**ppo_kwargs,
                                    cl_reg_coef=args.cl_reg_coef,
                                    expert_buffer_size=args.expert_buffer_size,
                                    replay_buffer_size=args.replay_buffer_size,
                                    env_seq=env_seq,
                                    nepochs_offline=args.nepochs_offline,
                                    bc_kl=args.bc_kl, distill_kl=args.distill_kl)
    
    


    return algo