"""MTSAC implementation based on Metaworld. Benchmarked on MT1.

This experiment shows how MTSAC adapts to 50 environents of the same type
but each environment has a goal variation.

https://arxiv.org/pdf/1910.10897.pdf
"""

import metaworld
from args import parse_args
from utils import set_seed, make_log_name

from garage.envs import normalize
from garage.experiment import CLTaskSampler
from garage.torch import set_gpu_mode
from garage.trainer import Trainer
from dm_control import suite

from garage.algo_factory import get_algo

import os
import wandb
from sklearn.utils import shuffle

args = parse_args()

MT50_ENV_SEQS = [
    'assembly-v2',                  'basketball-v2',        'bin-picking-v2',       'box-close-v2',             'button-press-topdown-v2',  # 0 1 2 3 4
    'button-press-topdown-wall-v2', 'button-press-v2',      'button-press-wall-v2', 'coffee-button-v2',         'coffee-pull-v2',           # 5 6 7 8 9
    'coffee-push-v2',               'dial-turn-v2',         'disassemble-v2',       'door-close-v2',            'door-lock-v2',             # 10 11 12 13 14
    'door-open-v2',                 'door-unlock-v2',       'drawer-close-v2',      'drawer-open-v2',           'faucet-open-v2',           # 15 16 17 18 19
    'faucet-close-v2',              'hammer-v2',            'hand-insert-v2',       'handle-press-side-v2',     'handle-press-v2',          # 20 21 22 23 24
    'handle-pull-side-v2',          'handle-pull-v2',       'lever-pull-v2',        'peg-insert-side-v2',       'peg-unplug-side-v2',       # 25 26 27 28 29
    'pick-out-of-hole-v2',          'pick-place-v2',        'pick-place-wall-v2',   'plate-slide-back-side-v2', 'plate-slide-back-v2',      # 30 31 32 33 34
    'plate-slide-side-v2',          'plate-slide-v2',       'push-back-v2',         'push-v2',                  'push-wall-v2',             # 35 36 37 38 39
    'reach-v2',                     'reach-wall-v2',        'shelf-place-v2',       'soccer-v2',                'stick-pull-v2',            # 40 41 42 43 44
    'stick-push-v2',                'sweep-into-v2',        'sweep-v2',             'window-close-v2',          'window-open-v2',           # 45 46 47 48 49
    ]

DMC_ENV_SEQS = [
    'DMControl-ball_in_cup-catch', 'DMControl-cartpole-balance', 'DMControl-cartpole-swingup', 'DMControl-finger-turn_easy', 'DMControl-fish-upright', 'DMControl-point_mass-easy', 'DMControl-reacher-easy', 
    ]

# faucet-open (19) --> push (38) --> sweep (47) --> button-press-topdown (4) —> window-open (49) --> sweep-into (46) --> button-press-wall (7) --> push-wall (39)
HARD_SEQ = [19, 38, 47, 4, 49, 46, 7, 39]
# faucet-open (19) -—> door-close (13) -—> button-press-topdown-wall (5) -—> handle-pull (26) -—> window-close (48) -—> plate-slide-back-side (33) -—> handle-press (24) -—> door-lock (14)
EASY_SEQ = [19, 13, 5, 26, 48, 33, 24, 14]
# door-unlock (16) --> faucet-open (19) --> handle-press-side (23) --> handle-pull-side (25) --> plate-slide-back-side (33) --> plate-slide-side (35) --> shelf-place (42) --> window-close (48)
RAND_SEQ = [16, 19, 23, 25, 33, 35, 42, 48] # Should be shuffled

if args.task_seq_idx is not None:
    if args.env_type == 'metaworld':
        env_seq = [MT50_ENV_SEQS[i] for i in args.task_seq_idx]
        task_seq_idx = args.task_seq_idx
    
    elif args.env_type == 'dm_control':
        env_seq = [DMC_ENV_SEQS[i] for i in args.task_seq_idx]
        task_seq_idx = args.task_seq_idx

else:
    if args.env_type == 'metaworld':
        if args.long_task_seq == 'hard':
            env_seq = [MT50_ENV_SEQS[i] for i in HARD_SEQ]
            task_seq_idx = HARD_SEQ
        elif args.long_task_seq == 'easy':
            env_seq = [MT50_ENV_SEQS[i] for i in EASY_SEQ]
            task_seq_idx = EASY_SEQ
        elif args.long_task_seq == 'rand':
            RAND_SEQ = shuffle(RAND_SEQ, random_state=args.seed)
            env_seq = [MT50_ENV_SEQS[i] for i in RAND_SEQ]
            task_seq_idx = RAND_SEQ

print(env_seq)


if args.wandb:
    wandb.login()
    wandb.init(
        name = args.proc_name,
        config = vars(args),
        project = 'Reset-Distill',
        dir = 'logs/'
    )

START_STEPS = 0
if args.rl_method == 'sac':
    START_STEPS = int(1e4)
_steps_per_task = args.steps_per_task
_gpu = args.device_type

# Set seed
seed = args.seed
set_seed(seed)
specs = None

if args.env_type == 'metaworld':
    
    mt50 = metaworld.MT50(seed=args.seed)
    total_steps = _steps_per_task+START_STEPS if args.rl_method == 'sac' else _steps_per_task
    print(args.use_exploration)
    task_sampler = CLTaskSampler(mt50, total_steps, seed, env_type='metaworld', wrapper=lambda env, _: normalize(env), exploration_steps= int(1e4) if args.use_exploration else 0)
    n_tasks = len(task_seq_idx)
    train_envs, test_envs = task_sampler.sample(task_seq_idx)

    env = test_envs[0]()
    test_envs = [env_up() for env_up in test_envs]

elif args.env_type == 'dm_control':
    task_sampler = CLTaskSampler(None, _steps_per_task+START_STEPS, seed, env_type='dm_control', wrapper=lambda env: normalize(env))
    n_tasks = len(task_seq_idx)
    train_envs, test_envs = task_sampler.sample(task_seq_idx)
    env = test_envs[0]
    specs = [test_env.spec for test_env in test_envs]

trainer = Trainer(None)

if args.rl_method == 'sac':
    timesteps = _steps_per_task * n_tasks
    
    batch_size = 500
    if args.env_type == 'dm_control':
        batch_size = 1000
    num_evaluation_steps = args.num_evaluation_steps
    epoch_cycles = num_evaluation_steps // batch_size
    epochs = timesteps // (batch_size * epoch_cycles)

elif args.rl_method == 'ppo':
    batch_size = 15000

    if args.env_type == 'dm_control':
        batch_size = 10000
    epochs = _steps_per_task // batch_size
    epochs *= len(env_seq)
    epoch_cycles = 0 # We do not use this

# set_gpu_mode before get_algo
if _gpu is not None:
    set_gpu_mode(True, _gpu)

algo = get_algo(
    args = args,
    spec = specs if args.env_type == 'dm_control' else env.spec, 
    n_tasks = n_tasks, 
    train_envs = train_envs, 
    test_envs = test_envs, 
    train_info = (batch_size, epoch_cycles),
    env_seq=env_seq,
    )

algo.to()
trainer.setup(algo=algo, env=train_envs)
trainer.train(n_epochs=epochs, batch_size=batch_size)

log_name = make_log_name(env_seq, args) 

if len(env_seq) == 1:
    if (args.first_task is None) or (env_seq[0] == args.first_task):
        algo.save_models(log_name=log_name)
        if args.rl_method == 'sac':
            algo.save_buffers(log_name=log_name)
        algo.save_rollouts(log_name=log_name)
        algo.save_results(log_name=log_name)
