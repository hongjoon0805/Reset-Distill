import numpy as np
import torch


from garage.torch.algos import MTSAC, PPO
from garage.replay_buffer import ExpertBuffer
from garage.torch import as_torch_dict
from garage import obtain_evaluation_episodes, EpisodeBatch

import wandb

class BC_SAC(MTSAC):
    def __init__(self, cl_reg_coef=1.0, expert_buffer_size=10000, **sac_kwargs):
        super().__init__(**sac_kwargs)

        self._cl_reg_coef=cl_reg_coef
        self._expert_buffer_size=expert_buffer_size
        self._expert_buffer=ExpertBuffer(capacity=expert_buffer_size, per_task=self._multi_input)
        self.BATCH_SIZE=64
        self._num_task_seen=0

    def on_task_start(self, seq_idx):

        assert self._num_task_seen == seq_idx

        samples = self.replay_buffer.sample_transitions(self._expert_buffer_size)
        samples = as_torch_dict(samples)

        obs = samples['observation']
        self._expert_buffer.add_observation_batch(obs, self.policy, seq_idx)

        self._num_task_seen += 1
    
    def cl_reg_loss(self, seq_idx):
        if seq_idx == 0:
            return 0
        
        if self._multi_input:
            loss = 0
            for task_idx in range(self._num_task_seen):
                obs, target_means, target_log_stds = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE, task_idx=task_idx)
                action_info = self.policy(obs, task_idx)[1]
                mus, log_stds = action_info['mean'], action_info['log_std']

                loss += self.kl_loss((mus, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef

                
        else:
            obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE)
            action_info = self.policy(obs, task_idxs)[1]
            mus, log_stds = action_info['mean'], action_info['log_std']

            loss = self.kl_loss((mus, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef

        return loss

    @staticmethod
    def kl_loss(dist1, dist2):
        """
        Computes KL(dist1 || dist2)
        """
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2

        std1 = log_std1.exp()
        std2 = log_std2.exp()

        mu_loss = 0.5 * ((1/(std2**2)) * (mu1-mu2).pow(2)).sum(dim=-1).mean()
        std_loss = 0.5 * (((std1/std2)**2 - 2*(log_std1 - log_std2))).sum(dim=-1).mean()

        return mu_loss + std_loss
        
    
class BC_PPO(PPO):

    def __init__(self, cl_reg_coef=1.0, expert_buffer_size=10000, **ppo_kwargs):
        super().__init__(**ppo_kwargs)

        self._cl_reg_coef=cl_reg_coef
        self._expert_buffer_size=expert_buffer_size
        self._expert_buffer=ExpertBuffer(capacity=expert_buffer_size, per_task=self._multi_input)
        self.BATCH_SIZE=64
        self._num_task_seen=0

    def on_task_start(self, seq_idx):

        assert self._num_task_seen == seq_idx

        eval_env = self._eval_env[seq_idx]
        len_cnt = 0
        episodes_list = []

        while True:
            
            episodes = obtain_evaluation_episodes(
                        self.policy,
                        eval_env,
                        seq_idx,
                        self._max_episode_length_eval,
                        num_eps=1,
                        deterministic=self._use_deterministic_evaluation)
            
            episodes_list.append(episodes)
            len_cnt += len(episodes.observations)
            if len_cnt >= self._expert_buffer_size:
                break
        
        episodes = EpisodeBatch.concatenate(*episodes_list)
        episodes_dict = {}
        episodes_dict['observation'] = episodes.observations[:self._expert_buffer_size]

        samples = as_torch_dict(episodes_dict)

        obs = samples['observation']
        self._expert_buffer.add_observation_batch(obs, self.policy, seq_idx)

        self._num_task_seen += 1
    
    def cl_reg_loss(self, seq_idx):
        if seq_idx == 0:
            return 0
        
        if self._multi_input:
            loss = 0
            for task_idx in range(self._num_task_seen):
                obs, target_means, target_log_stds = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE, task_idx=task_idx)
                action_info = self.policy(obs, task_idx)[1]
                mus, log_stds = action_info['mean'], action_info['log_std']

                loss += self.kl_loss((mus, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef

                
        else:
            obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE)
            action_info = self.policy(obs, task_idxs)[1]
            mus, log_stds = action_info['mean'], action_info['log_std']

            loss = self.kl_loss((mus, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef

        return loss

    @staticmethod
    def kl_loss(dist1, dist2):
        """
        Computes KL(dist1 || dist2)
        """
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2

        std1 = log_std1.exp()
        std2 = log_std2.exp()

        mu_loss = 0.5 * ((1/(std2**2)) * (mu1-mu2).pow(2)).sum(dim=-1).mean()
        std_loss = 0.5 * (((std1/std2)**2 - 2*(log_std1 - log_std2))).sum(dim=-1).mean()

        return mu_loss + std_loss
        
