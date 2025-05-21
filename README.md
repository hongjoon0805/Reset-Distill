# Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

---

This repository contains official code implementation of "Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline".

Our code implementation is designed for running 2-tasks and long task sequences experiments to reproduce the results in our [paper](https://openreview.net/forum?id=KAIqwkB3dT). 

We built our code implementation based on the [garage](https://github.com/rlworkgroup/garage) repository.

---

## Environments

Before running the experiment, you should set up the environment.

Please run the following command.

```
conda env create --file environment.yaml
```

---

## Running experiments

The basic form of the execution command is as follows:

```
python3 main_garage.py --cl_method [cl-method] --rl_method [rl-method] --env_type [environment type] --task_seq_idx [sequence of tasks] --wandb [True or False] --seed [seed] --device_type [device num] --proc_name [your process name]
```

Available options:

- cl-method: `bc`, `ewc`, `finetuning`, `pandc`, `rnd`
- rl-method: `sac`, `ppo`
- environment type: `metaworld`, `dm_control`
- sequence of tasks: sequence of task ids in `metaworld` or `dm_control`

If you want to run the long task sequence experiment, please use `--long_task_seq [sequence type]` (sequence type: `hard`, `easy`, `random`) instead of `--task_seq_idx`.

---

### Singe task experiment

Before running the 2-tasks experiment or R&D, we recommend running singe task experiments first to save the models and rollouts.

The execution command is as follows:

- Meta-World
```
# SAC (button-press-v2)
python3 main_garage.py --cl_method finetuning --rl_method sac --env_type metaworld --task_seq_idx 6 --seed 0 --proc_name Single_task_metaworld_SAC_button-press-v2

# PPO (button-press-v2)
python3 main_garage.py --cl_method finetuning --rl_method ppo --env_type metaworld --task_seq_idx 6 --seed 0 --proc_name Single_task_metaworld_PPO_button-press-v2
```

- DeepMind Control
```
# SAC (DMControl-cartpole-swingup)
python3 main_garage.py --cl_method finetuning --rl_method sac --env_type metaworld --task_seq_idx 2 --seed 0 --proc_name Single_task_SAC_DMControl-cartpole-swingup

# PPO (DMControl-cartpole-swingup)
python3 main_garage.py --cl_method finetuning --rl_method ppo --env_type metaworld --task_seq_idx 2 --seed 0 --proc_name Single_task_PPO_DMControl-cartpole-swingup
```

By replacing the task numbers in `--task_seq_idx`, you can run experiments on other tasks such as `faucet-open-v2` or `sweep-into-v2`.

---

### 2-tasks experiment

For running the 2-tasks experiment, the execution command is as follows:

- Meta-World
```
# SAC (button-press-v2 --> button-press-wall-v2)
python3 main_garage.py --cl_method finetuning --rl_method sac --env_type metaworld --first_task button-press-v2 --task_seq_idx 7 --seed 0 --proc_name Two_tasks_metaworld_SAC_button-press-wall-v2_button-press-wall-v2

# PPO (button-press-v2 --> button-press-wall-v2)
python3 main_garage.py --cl_method finetuning --rl_method ppo --env_type metaworld --first_task button-press-v2 --task_seq_idx 7 --seed 0 --proc_name Two_tasks_metaworld_PPO_button-press-wall-v2_button-press-wall-v2
```

- DeepMind Control
```
# SAC (DMControl-cartpole-swingup --> DMControl-ball_in_cup-catch)
python3 main_garage.py --cl_method finetuning --rl_method sac --env_type metaworld --first_task DMControl-cartpole-swingup --task_seq_idx 0 --seed 0 --proc_name Two_tasks_metaworld_SAC_DMControl-cartpole-swingup_DMControl-ball_in_cup-catch

# PPO (DMControl-cartpole-swingup --> DMControl-ball_in_cup-catch)
python3 main_garage.py --cl_method finetuning --rl_method ppo --env_type metaworld --first_task DMControl-cartpole-swingup --task_seq_idx 0 --seed 0 --proc_name Two_tasks_metaworld_PPO_DMControl-cartpole-swingup_DMControl-ball_in_cup-catch
```

By modifying the `--first_task` and `--task_seq_idx`, you can run experiments on different task pairs.

---

### Long task sequence experiment

For running the long task sequence experiment, the execution command is as follows:

- Meta-World (Hard sequence)

```
# SAC w/ Finetuning
python3 main_garage.py --cl_method finetuning --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_finetuning

# SAC w/ P&C
python3 main_garage.py --cl_method pandc --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_pandc

# SAC w/ EWC
python3 main_garage.py --cl_method ewc --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_ewc

# SAC w/ ClonEx (BC)
python3 main_garage.py --cl_method bc --use_exploration True --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_ClonEx

# SAC w/ ClonEx (BC) + CReLU
python3 main_garage.py --cl_method bc --use_exploration True --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_ClonEx_CReLU

# SAC w/ ClonEx (BC) + InFeR
python3 main_garage.py --cl_method bc --use_exploration True --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_ClonEx_InFeR

# SAC w/ R&D
python3 main_garage.py --cl_method rnd --rl_method sac --env_type metaworld --long_task_seq --seed 0 --proc_name Hard_sequence_experiment_SAC_RND
```
> [!CAUTION]
> Before running R&D, you should prepare the models and rollouts by running the single task experiment.

If you want to run the experiments using PPO, please modify `--rl_method` as `ppo`

---

## Citation

```bibtex
@inproceedings{
ahn2025prevalence,
title={Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline},
author={Hongjoon Ahn and Jinu Hyeon and Youngmin Oh and Bosun Hwang and Taesup Moon},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=KAIqwkB3dT}
}

```
