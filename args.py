import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Continual RL with Continual World / DMLab')

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--task_seq_idx', type=int, default=None, nargs='+', metavar='N',
                        help='task sequence index for CL (default: 0)')
    
    task_group.add_argument('--long_task_seq', type=str, default=None, metavar='N',
                            choices=['hard', 'easy', 'rand'],
                        help='task sequence index for CL (default: 0)')
    
    parser.add_argument('--env_type', default='', required=True, type=str,
                        choices=['metaworld', 'dm_control'], 
                        help='Environment for CL')
    parser.add_argument('--rl_method', default='', required=True, type=str,
                        choices=['ppo', 'sac',], 
                        help='Environment for CL')
    parser.add_argument('--cl_method', default='', type=str, required=True,
                        choices=['bc', 
                                 'ewc',
                                 'finetuning',
                                 'rnd', 
                                 'pandc',], 
                        help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='seed (default: 0)')
    
    parser.add_argument('--wandb', type=str2bool, default=True, metavar='N',
                        help='Use wandb')
    
    parser.add_argument('--cl_reg_coef', type=float, default=1.0, metavar='N',
                        help='regularization strenghth for cl methods (default: 1.0)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='N',
                        help='total number of tasks (default: 10)')
    parser.add_argument('--steps_per_task', type=int, default=int(3e6), metavar='N',
                        help='steps per task (default: int(3e6))')
    parser.add_argument('--nepochs_offline', type=int, default=int(5), metavar='N',
                        help='number of epochs for offline training (default: int(100))')

    parser.add_argument('--num_evaluation_steps', type=int, default=int(1e5), metavar='N',
                        help='the number of steps on the interval between evaluation (default: int(1e5))')
    parser.add_argument('--num_evaluation_episodes', type=int, default=50, metavar='N',
                        help='the number of episodes for evaluation (default: 50)')
    parser.add_argument('--expert_buffer_size', type=int, default=10000, metavar='N',
                        help='size of expert buffer for BC (default: 10000)')
    
    # Arguments for PnC
    parser.add_argument('--compress_step', default=int(1e6), type=int, help="(default: 1e6)")
    parser.add_argument('--use_pandc_bc', type=str2bool, default=False, metavar='N',
                        help='Use BC in P&C')
    parser.add_argument('--reset_column', type=str2bool, default=False, metavar='N',
                        help='Reset column in P&C')
    parser.add_argument('--reset_adaptor', type=str2bool, default=False, metavar='N',
                        help='Reset adaptor in P&C')
    parser.add_argument('--zero_alpha', type=str2bool, default=False, metavar='N',
                        help='Set alpha to zero in P&C adaptor')
    
    
    parser.add_argument('--use_exploration', type=str2bool, default=False, metavar='N',
                        help='Use exploration technique')
    parser.add_argument('--q_reset', type=str2bool, default='False', metavar='N',
                        help='Reset Q function')
    parser.add_argument('--policy_reset', type=str2bool, default='False', metavar='N',
                        help='Reset policy')
    parser.add_argument('--first_task', type=str, default=None, metavar='N',
                        help='Name of first task')
    parser.add_argument('--bc_kl', default='reverse', type=str,
                        choices=['forward', 'reverse'], 
                        help='The direction of the behaviour cloning loss')
    parser.add_argument('--distill_kl', default='forward', type=str,
                        choices=['forward', 'reverse'], 
                        help='The direction of the distillation loss')
    parser.add_argument('--device_type', default=None, type=int,
                        choices=[None, 0, 1, 2, 3, 4, 5, 6, 7], 
                        help='None: use cpu, 0~3: use GPU with device number #')
    parser.add_argument('--proc_name', default='Negative-Transfer-Test', type=str, required=True,
                        help='(default=%(default)s)')
    parser.add_argument('--replay_buffer_size', default=int(1e6), type=int,
                        help="R&D buffer size")
    parser.add_argument('--replay_k', default=5, type=int,
                        help='Number of HER transitions to add for each regular transition')
    parser.add_argument('--infer', default=False, type=str2bool,
                        help="Use InFeR loss if True")
    parser.add_argument('--ReDo', default=False, type=str2bool,
                        help="Use ReDo if True")
    parser.add_argument('--reset_offline_actor', default=False, type=str2bool,
                        help="Reset the offline actor in R&D")
    parser.add_argument('--wasserstein', default=0, type=float,
                        help="Use Wasserstein loss if positive")
    
    parser.add_argument('--fixed_alpha', default=None, type=float,
                        help='SAC alpha tuning')
    parser.add_argument('--lr_clip_range', default=0.2, type=float,
                        help='PPO likelihood-ratio clip range')


    parser.add_argument('--crelu', type=str2bool, default='False', metavar='N',
                        help='Use CReLU')
    
    parser.add_argument('--no_stats', type=str2bool, default=True, metavar='N',
                        help='Do not store the statistics')
    
    args = parser.parse_args()

    return args