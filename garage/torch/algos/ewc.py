import numpy as np
import torch


from garage.torch.algos import MTSAC, PPO
from garage.torch._functions import zero_optim_grads, np_to_torch
from garage.torch import as_torch_dict, global_device, state_dict_to, compute_advantages, filter_valids
from garage import obtain_evaluation_episodes, EpisodeBatch
import copy

import wandb

class EWC_SAC(MTSAC):

    def __init__(self, cl_reg_coef=1.0, **sac_kwargs):
        super().__init__(**sac_kwargs)
        self._cl_reg_coef=cl_reg_coef
        self.BATCH_SIZE=20
        self.SAMPLE_SIZE = 20000
        self._num_task_seen=0
        self.policy_old = copy.deepcopy(self.policy)
        self._fisher = {}

    def cl_reg_loss(self, seq_idx):

        if seq_idx == 0:
            return 0

        loss = 0
        for (name, p), (_, p_old) in zip(self.policy.named_parameters(), self.policy_old.named_parameters()):
            loss += (self._fisher[name] * (p-p_old).pow(2)).sum() * self._cl_reg_coef
        
        return loss

    def on_task_start(self, seq_idx):

        assert self._num_task_seen == seq_idx

        if seq_idx == 0:
            for name, p in self.policy.named_parameters():
                self._fisher[name] = torch.zeros_like(p, device=global_device())


        self.update_fisher_matrix(seq_idx)
        policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.policy_old.load_state_dict(policy_state_dict)
        self.policy_old.to(device=global_device())

        self._num_task_seen += 1
        

    def update_fisher_matrix(self, seq_idx):

        new_fisher = {}
        for name, p in self.policy.named_parameters():
            new_fisher[name] = torch.zeros_like(p)
        
        total_samples = self.replay_buffer.sample_transitions(self.SAMPLE_SIZE)
        total_samples = as_torch_dict(total_samples)

        total_obs = total_samples['observation']

        for i in range(0, self.SAMPLE_SIZE, self.BATCH_SIZE):
            start = i
            end = i+self.BATCH_SIZE
            if i+self.BATCH_SIZE > self.SAMPLE_SIZE:
                end = self.SAMPLE_SIZE
            
            obs = total_obs[start:end]

            samples_data = {}
            samples_data['observation'] = obs

            action_dists = self.policy(obs, seq_idx)[0]
            new_actions_pre_tanh, new_actions = (
                action_dists.rsample_with_pre_tanh_value())
            log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh)

            loss = self._actor_objective(samples_data, new_actions, log_pi_new_actions, seq_idx=seq_idx)
            zero_optim_grads(self._policy_optimizer)
            loss.backward()
            
            for name, p in self.policy.named_parameters():
                if p.grad is not None:
                    new_fisher[name] += (p.grad.data * self.BATCH_SIZE).pow(2)

        for name, p in self.policy.named_parameters():
            self._fisher[name] = ((new_fisher[name] / self.SAMPLE_SIZE) + self._fisher[name] * seq_idx) / (seq_idx + 1)
        

class EWC_PPO(PPO):
    def __init__(self, cl_reg_coef=1.0, **ppo_kwargs):
        super().__init__(**ppo_kwargs)
        
        self._cl_reg_coef=cl_reg_coef
        self.BATCH_SIZE=128
        self.SAMPLE_SIZE = 20000
        self._num_task_seen=0
        self.policy_old = copy.deepcopy(self.policy)
        self._fisher = {}

    def cl_reg_loss(self, seq_idx):

        if seq_idx == 0:
            return 0

        loss = 0
        for (name, p), (_, p_old) in zip(self.policy.named_parameters(), self.policy_old.named_parameters()):
            loss += (self._fisher[name] * (p-p_old).pow(2)).sum() * self._cl_reg_coef
        
        return loss

    def on_task_start(self, seq_idx):

        assert self._num_task_seen == seq_idx

        if seq_idx == 0:
            for name, p in self.policy.named_parameters():
                self._fisher[name] = torch.zeros_like(p, device=global_device())


        self.update_fisher_matrix(seq_idx)
        policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.policy_old.load_state_dict(policy_state_dict)
        self.policy_old.to(device=global_device())

        self._num_task_seen += 1
        

    def update_fisher_matrix(self, seq_idx):

        new_fisher = {}
        for name, p in self.policy.named_parameters():
            new_fisher[name] = torch.zeros_like(p)
        

        batch_size = 2000
        steps = self.SAMPLE_SIZE // batch_size

        for _ in range(steps):
            eps_lst = []
            sample_count = 0
            while sample_count < batch_size:
                es = obtain_evaluation_episodes(
                    self.policy,
                    self._eval_env[seq_idx],
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps = 1,
                    deterministic = self._use_deterministic_evaluation
                )
                eps_lst.append(es)
                sample_count += len(es.observations)
                
            eps = EpisodeBatch.concatenate(*eps_lst)
            obs = np_to_torch(eps.padded_observations)
            rewards = np_to_torch(eps.padded_rewards)
            valids = eps.lengths
            with torch.no_grad():
                baselines = self._value_function(obs, seq_idx=seq_idx)
            
            self._episode_reward_mean.append(rewards.mean().item())

            if self._maximum_entropy:
                policy_entropies = self.policy(obs, seq_idx)[0].entropy()
                rewards += self._policy_ent_coeff * policy_entropies

            obs_flat = np_to_torch(eps.observations)
            actions_flat = np_to_torch(eps.actions)
            rewards_flat = np_to_torch(eps.rewards)
            advs_flat = self._compute_advantage(rewards, valids, baselines)

            loss = 0
            for (ob, ac, re, ad) in self._policy_optimizer.get_minibatch(
                obs_flat, actions_flat, rewards_flat, advs_flat
            ):
                loss = self._compute_loss_with_adv(ob, ac, re, ad, seq_idx)
                zero_optim_grads(self._policy_optimizer._optimizer)
                loss.backward()
                
                for name, p in self.policy.named_parameters():
                    if p.grad is not None:
                        new_fisher[name] += (p.grad.data * self.BATCH_SIZE).pow(2)


        for name, p in self.policy.named_parameters():
            self._fisher[name] = ((new_fisher[name] / self.SAMPLE_SIZE) + self._fisher[name] * seq_idx) / (seq_idx + 1)
