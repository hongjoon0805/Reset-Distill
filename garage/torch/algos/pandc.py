import numpy as np
import abc
import torch
from copy import deepcopy

from garage import EpisodeBatch, log_multitask_performance, obtain_evaluation_episodes, log_performance
from garage.torch import as_torch_dict, global_device, state_dict_to
from garage.torch.algos import MTSAC, PPO
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.torch.optimizers import OptimizerWrapper
from garage.replay_buffer import ExpertBuffer

class P_and_C(abc.ABC):

    def __init__(self, cl_reg_coef, compress_step, bc = False, reset_column=False, reset_adaptor=False):
        self._cl_reg_coef = cl_reg_coef
        self._compress_step = compress_step
        self._bc = bc
        self._reset_column = reset_column
        self._reset_adaptor = reset_adaptor

        if reset_column:
            print('We will reset the active column when we change the task')
        if reset_adaptor:
            print('We will reset the adaptor when we change the task')

        if bc:
            expert_buffer_size = 10000
            self._expert_buffer_size=expert_buffer_size
            self._expert_buffer=ExpertBuffer(capacity=expert_buffer_size)

        self.device = global_device()

        self._num_task_seen = 0

        self._fisher = dict()
        for name, p in self.policy.named_parameters():
            self._fisher[name] = torch.zeros_like(p, device = self.device)

        self.generate_kb()
    
    @abc.abstractmethod
    def generate_kb(self):
        pass

    def ewc_loss(self, seq_idx):
        if seq_idx == 0: return 0
        
        loss = 0
        for (name, p), (_, p_old) in zip(self.policy_kb.named_parameters(), self.policy_kb_prev.named_parameters()):
            loss += (self._fisher[name] * (p - p_old).pow(2)).sum() * self._cl_reg_coef
            
        return loss
    
    @abc.abstractmethod
    def _update_fisher_matrix(self, seq_idx):
        pass

    @abc.abstractmethod
    def _compress(self, seq_idx):
        pass

    @staticmethod
    def kl_loss(dist1, dist2):
        mu1, log_std1 = dist1['mean'], dist1['log_std']
        mu2, log_std2 = dist2['mean'], dist2['log_std']
        
        std1 = log_std1.exp()
        std2 = log_std2.exp()
        
        mu_loss = .5 * ((1/(std2**2)) * (mu1 - mu2).pow(2)).sum(dim=-1).mean()
        std_loss = .5 * (((std1/std2)**2 - 2*(log_std1 - log_std2))).sum(dim=-1).mean()

        return mu_loss + std_loss

    def _compress_loss(self, seq_idx, obs):
        dist = self.policy_kb(obs, seq_idx)[1]
        with torch.no_grad():
            target_dist = self.policy(obs, seq_idx)[1]

        cl_reg_loss = 0

        if seq_idx > 0:
            if self._bc:
                batch_size = len(obs)
                buffer_obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(batch_size)
                action_info = self.policy_kb(buffer_obs, task_idxs)[1]
                mus, log_stds = action_info['mean'], action_info['log_std']

                dist1 = {'mean': mus, 'log_std': log_stds}
                dist2 = {'mean': target_means, 'log_std': target_log_stds}
                
                cl_reg_loss = self._cl_reg_coef * self.kl_loss(dist1, dist2)
            else:
                cl_reg_loss = self._cl_reg_coef * self.ewc_loss(seq_idx)
        
        return cl_reg_loss + self.kl_loss(dist, target_dist)


class P_and_C_SAC(MTSAC, P_and_C):

    def __init__(self, cl_reg_coef: float = 1.0, compress_step: int = 1e4, bc = False, reset_column=False, reset_adaptor=False, **sac_kwargs):

        MTSAC.__init__(self, **sac_kwargs)
        P_and_C.__init__(self,
                         cl_reg_coef = cl_reg_coef,
                         compress_step = compress_step,
                         bc=bc, 
                         reset_column=reset_column, 
                         reset_adaptor=reset_adaptor)
    
    def generate_kb(self):
        self.policy_kb = deepcopy(self.policy).to(self.device)
        self._policy_kb_optimizer = self._optimizer(self.policy_kb.parameters(), lr = self._policy_lr)
        self._policy_kb_optimizer.load_state_dict(state_dict_to(self._policy_kb_optimizer.state_dict(), self.device))
        self.policy_kb_prev = deepcopy(self.policy_kb).to(self.device)
        

    def _update_fisher_matrix(self, seq_idx):

        print(f"Updating fisher matrix: seq_idx = {seq_idx}")

        new_fisher = dict()    
        for name, p in self.policy_kb.named_parameters():
            new_fisher[name] = torch.zeros_like(p).to(self.device)

        batch_size = 20
        sample_size = 20000
        
        total_samples = self.replay_buffer.sample_transitions(sample_size)
        total_samples = as_torch_dict(total_samples)
        total_obs = total_samples['observation']

        for i in range(0, sample_size, batch_size):
            start = i
            end = i+batch_size
            if i+batch_size > sample_size:
                end = sample_size
            
            obs = total_obs[start:end]

            samples_data = {}
            samples_data['observation'] = obs

            action_dists = self.policy_kb(obs, seq_idx)[0]
            new_actions_pre_tanh, new_actions = (
                action_dists.rsample_with_pre_tanh_value())
            log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh)

            loss = self._actor_objective(samples_data, new_actions, log_pi_new_actions, seq_idx=seq_idx)
            zero_optim_grads(self._policy_kb_optimizer)
            loss.backward()
            
            for name, p in self.policy_kb.named_parameters():
                if p.grad is not None:
                    new_fisher[name] += (p.grad.data * batch_size).pow(2)
                    
        for name, p in self.policy_kb.named_parameters():
            self._fisher[name] = ((new_fisher[name] / sample_size) + self._fisher[name] * seq_idx) / (seq_idx + 1)

        print("Fisher matrix updated!")

    def _compress(self, seq_idx):

        print(f"Compress step: seq_idx = {seq_idx}")

        steps = self._compress_step // self._buffer_batch_size
        for _ in range(steps):
            
            samples_data = self.replay_buffer.sample_transitions(self._buffer_batch_size)
            samples_data = as_torch_dict(samples_data)
            
            obs = samples_data['observation']
            
            loss = self._compress_loss(seq_idx, obs)
            
            zero_optim_grads(self._policy_kb_optimizer)
            
            loss.backward()
            self._policy_kb_optimizer.step()

        print("Compress step done!")
    
    def on_task_start(self, seq_idx):
        
        assert self._num_task_seen == seq_idx
        
        self._compress(seq_idx)
        if self._bc:
            samples = self.replay_buffer.sample_transitions(self._expert_buffer_size)
            samples = as_torch_dict(samples)

            obs = samples['observation']
            self._expert_buffer.add_observation_batch(obs, self.policy, seq_idx)

        else:
            self._update_fisher_matrix(seq_idx)
        
        policy_kb_state_dict = self.policy_kb._module.state_dict()
        self.policy_kb_prev._module.load_state_dict(policy_kb_state_dict)

        if self._reset_column:
            
            self.policy.reset_parameter(adaptor=self._reset_adaptor)
            print('Reset the active column')
            if self._reset_adaptor:
                print('Reset the adaptor')
        self._num_task_seen += 1

    def _evaluate_policy(self, epoch):
        eval_eps = []
        for seq_idx, eval_env in enumerate(self._eval_env):

            self.on_test_start(seq_idx)

            # Use policy_kb at evaluation step only when current idx > seq idx
            if self.seq_idx > seq_idx:
                eps = obtain_evaluation_episodes(
                    self.policy_kb,
                    eval_env,
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
                
            
            else:
                eps = obtain_evaluation_episodes(
                    self.policy,
                    eval_env,
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
            
            eval_eps.append(eps)
            self.on_test_end(seq_idx)

            if isinstance(self.env_spec, list):
                last_return = log_performance(epoch,
                                      eps,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)

        if not isinstance(self.env_spec, list):
            eval_eps = EpisodeBatch.concatenate(*eval_eps)
            last_return = log_multitask_performance(epoch, eval_eps,
                                                    self._discount,
                                                    self.results,
                                                    use_wandb=self._use_wandb)
        
        
        return last_return
    
    def _get_policy_output(self, obs, seq_idx):
        if seq_idx == 0:
            kb_features = None
        else:
            with torch.no_grad():
                _ = self.policy_kb(obs, seq_idx)[0]
                kb_features = self.policy_kb._features

        action_dists = self.policy(obs, seq_idx, features = kb_features)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        
        return action_dists, new_actions, log_pi_new_actions
    

class P_and_C_PPO(PPO, P_and_C):

    def __init__(self, cl_reg_coef: float = 1.0, compress_step: int = 1e4, bc = False, reset_column=False, reset_adaptor=False, **ppo_kwargs):
        PPO.__init__(self, **ppo_kwargs)
        P_and_C.__init__(self,
                         cl_reg_coef = cl_reg_coef,
                         compress_step = compress_step,
                         bc = bc,
                         reset_column=reset_column, 
                         reset_adaptor=reset_adaptor)
        self.generate_kb()

    def generate_kb(self):
        self.policy_kb = deepcopy(self.policy).to(self.device)
        self._policy_kb_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=5e-4)),
                self.policy_kb,
                max_optimization_epochs=32,
                minibatch_size=128)
        self._policy_kb_optimizer._optimizer.load_state_dict(
            state_dict_to(self._policy_kb_optimizer._optimizer.state_dict(), self.device)
            )
        self.policy_kb_prev = deepcopy(self.policy_kb).to(self.device)
    
    def _update_fisher_matrix(self, seq_idx):
        print(f"Updating fisher matrix: seq_idx = {seq_idx}")

        new_fisher = dict()    
        for name, p in self.policy_kb.named_parameters():
            new_fisher[name] = torch.zeros_like(p).to(self.device)

        batch_size = 2000
        sample_size = 20000
        steps = sample_size // batch_size

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
                policy_entropies = self.policy_kb(obs, seq_idx)[0].entropy()
                rewards += self._policy_ent_coeff * policy_entropies

            obs_flat = np_to_torch(eps.observations)
            actions_flat = np_to_torch(eps.actions)
            rewards_flat = np_to_torch(eps.rewards)
            advs_flat = self._compute_advantage(rewards, valids, baselines)

            for (ob, ac, re, ad) in self._policy_optimizer.get_minibatch(
                obs_flat, actions_flat, rewards_flat, advs_flat
            ):
                loss = self._compute_loss_with_adv(ob, ac, re, ad, seq_idx)
                zero_optim_grads(self._policy_optimizer._optimizer)
                loss.backward()

                for name, p in self.policy_kb.named_parameters():
                    if p.grad is not None:
                        new_fisher[name] += (p.grad.data * self._policy_optimizer._minibatch_size).pow(2)
            
        for name, p in self.policy_kb.named_parameters():
            self._fisher[name] = ((new_fisher[name] / sample_size) + self._fisher[name] * seq_idx) / (seq_idx + 1)

        print("Fisher matrix updated!")

    def _compress(self, seq_idx):
        
        print(f"Compress step: seq_idx = {seq_idx}")

        batch_size = 2000
        steps = self._compress_step // batch_size

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
            obs = np_to_torch(eps.observations)

            loss = self._compress_loss(seq_idx, obs)

            zero_optim_grads(self._policy_kb_optimizer._optimizer)

            loss.backward()
            self._policy_kb_optimizer.step()

        print("Compress step done!")

    def on_task_start(self, seq_idx):
        
        assert self._num_task_seen == seq_idx
        
        self._compress(seq_idx)
        if self._bc:
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

        else:
            self._update_fisher_matrix(seq_idx)

        policy_kb_state_dict = self.policy_kb._module.state_dict()
        self.policy_kb_prev._module.load_state_dict(policy_kb_state_dict)

        if self._reset_column:
            
            self.policy.reset_parameter(adaptor=self._reset_adaptor)
            self._old_policy.reset_parameter(adaptor=self._reset_adaptor)
            print('Reset the active column')
            if self._reset_adaptor:
                print('Reset the adaptor')

        self._num_task_seen += 1

    def _evaluate_policy(self, epoch):
        eval_eps = []
        for seq_idx, eval_env in enumerate(self._eval_env):

            self.on_test_start(seq_idx)

            # Use policy_kb at evaluation step only when current idx > seq idx
            if self.seq_idx > seq_idx:
                eps = obtain_evaluation_episodes(
                    self.policy_kb,
                    eval_env,
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
                eval_eps.append(eps)
            
            else:
                eps = obtain_evaluation_episodes(
                    self.policy,
                    eval_env,
                    seq_idx,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    deterministic=self._use_deterministic_evaluation)
                eval_eps.append(eps)
            
            self.on_test_end(seq_idx)
            if isinstance(self._env_spec, list):
                last_return = log_performance(epoch,
                                      eps,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)

        if not isinstance(self._env_spec, list):
            eval_eps = EpisodeBatch.concatenate(*eval_eps)
            last_return = log_multitask_performance(epoch, eval_eps,
                                                    self._discount,
                                                    self.results, 
                                                    use_wandb=self._use_wandb)
        return last_return
    