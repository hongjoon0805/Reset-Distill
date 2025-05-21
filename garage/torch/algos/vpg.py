"""Vanilla Policy Gradient (REINFORCE)."""
import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage import (EpisodeBatch, log_multitask_performance,
                    obtain_evaluation_episodes)
from garage.np import discount_cumsum
from garage.np.algos import RLAlgorithm
from garage.torch import compute_advantages, filter_valids, as_torch_dict
from garage.torch._functions import np_to_torch, zero_optim_grads, feature_rank, weight_deviation, weight_hessian
from garage.torch import global_device, state_dict_to
from garage.torch.optimizers import OptimizerWrapper
from time import time

import wandb
import pickle
import os

def load_model(model, model_name, first_task, seed):
    # Load policy
    copied_model = copy.deepcopy(model)
    name = model_name.format(first_task, seed)
    loaded_state_dict = torch.load('./models/ppo_models/'+name, map_location=global_device())

    copied_model_state_dict = copied_model.state_dict()
    new_state_dict = {}

    for k in loaded_state_dict.keys():
        
        if loaded_state_dict[k].shape == copied_model_state_dict[k].shape:
            new_state_dict[k] = loaded_state_dict[k]
        else:
            new_state_dict[k] = copied_model_state_dict[k]

    copied_model_state_dict.update(new_state_dict)
    model.load_state_dict(copied_model_state_dict)

    return


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient (REINFORCE).
    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.
    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(
        self,
        env_spec,
        policy,
        value_function,
        sampler,
        seed = 0,
        policy_optimizer=None,
        vf_optimizer=None,
        num_train_per_epoch=1,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy',
        eval_env=None,
        num_evaluation_episodes=10,
        use_deterministic_evaluation=True,
        log_name=None,
        q_reset=False,
        policy_reset=False,
        first_task = None, 
        use_wandb=True,
        infer=False,
        crelu=False,
        wasserstein=0, 
        ReDo=False,
        no_stats=False, 
        multi_input=False):
        
        self._discount = discount
        self.policy = policy
        if isinstance(env_spec,list):
            max_episode_length = env_spec[0].max_episode_length
        else:
            max_episode_length = env_spec.max_episode_length
        self.max_episode_length = max_episode_length

        self._value_function = value_function
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._n_samples = num_train_per_epoch
        self._env_spec = env_spec
        
        self._eval_env = eval_env
        self._seed = seed
        self._num_evaluation_episodes = num_evaluation_episodes
        self._use_deterministic_evaluation = use_deterministic_evaluation
        self._max_episode_length_eval = max_episode_length

        self._value_reset = q_reset
        self._policy_reset = policy_reset
        self._first_task = first_task
        self._infer = infer
        self._wasserstein = (wasserstein > 0)
        self._ReDo = ReDo
        self._no_stats = no_stats
        self._multi_input = multi_input

        self._log_name=log_name
        self._use_wandb = use_wandb
        

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=10)
        self._sampler = sampler

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        else:
            self._policy_optimizer = OptimizerWrapper(torch.optim.Adam, policy)
        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        else:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam,
                                                  value_function)

        self._old_policy = copy.deepcopy(self.policy)

        self.global_step = 0
        self.seq_idx = 0
        self.start_time = time()
        self.begin = self.start_time
        self.results = {}

        self.results['Running avg. of episode return'] = []
        self.results['Policy loss'] = []
        self.results['Value loss'] = []
        self.results['KL'] = []
        self.results['Speed (it/s)'] = []

        if self._no_stats == False:
            self.results['Policy dormant ratio'] = []
            self.results['Value dormant ratio'] = []
            
            self.results['Policy feature rank'] = []
            self.results['Value feature rank'] = []
            self.results['Policy hessian rank'] = []
            self.results['Value hessian rank'] = []
            self.results['Policy weight change'] = []
            self.results['Value weight change'] = []

        self._random_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self._random_vf_state_dict = copy.deepcopy(self._value_function.state_dict())

        # For ReDo
        self.random_policy = copy.deepcopy(self.policy)
        self._random_value_function = copy.deepcopy(self._value_function)

        if infer:
            self._infer_target_policy = copy.deepcopy(self.policy)
            self._infer_target_vf = copy.deepcopy(self._value_function)
            self._infer_alpha = 1.
            self._infer_beta = 10.
            print("Use InFeR Loss")
        
        if wasserstein:
            self._wasserstein_target_policy = copy.deepcopy(self.policy)
            self._wasserstein_target_vf = copy.deepcopy(self._value_function)
            self._wasserstein_lambda = wasserstein
            print("Use Wasserstein Regularization")


        if self._first_task is not None:
            if 'DMC' in self._first_task:
                policy_name = 'policy_dm_control_ppo_{}_1000000_{}.pt'
                old_policy_name = 'old_policy_dm_control_ppo_{}_1000000_{}.pt'
                vf_name = 'vf_dm_control_ppo_{}_1000000_{}.pt'

            else:
                policy_name = 'policy_metaworld_ppo_{}_3000000_{}.pt'
                old_policy_name = 'old_policy_metaworld_ppo_{}_3000000_{}.pt'
                vf_name = 'vf_metaworld_ppo_{}_3000000_{}.pt'

                if crelu:
                    policy_name = 'policy_CReLU_metaworld_ppo_{}_3000000_{}.pt'
                    old_policy_name = 'old_policy_CReLU_metaworld_ppo_{}_3000000_{}.pt'
                    vf_name = 'vf_CReLU_metaworld_ppo_{}_3000000_{}.pt'

                if wasserstein:
                    policy_name = 'policy_Wasserstein_0.1_metaworld_ppo_{}_3000000_{}.pt'
                    old_policy_name = 'old_policy_Wasserstein_0.1_metaworld_ppo_{}_3000000_{}.pt'
                    vf_name = 'vf_Wasserstein_0.1_metaworld_ppo_{}_3000000_{}.pt'
            
            load_model(self.policy, policy_name, self._first_task, self._seed)
            load_model(self._old_policy, old_policy_name, self._first_task, self._seed)
            load_model(self._value_function, vf_name, self._first_task, self._seed)
            
        
        if self._value_reset:
            print('############################################################')
            print('                     Value-reset!!!!!                       ')
            print('############################################################')

            self._value_function.load_state_dict(self._random_vf_state_dict)
        
        if self._policy_reset:
            print('############################################################')
            print('                     Policy-reset!!!!!                      ')
            print('############################################################')
            self.policy.load_state_dict(self._random_policy_state_dict)



    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    @property
    def discount(self):
        """Discount factor used by the algorithm.
        Returns:
            float: discount factor.
        """
        return self._discount

    def _train_once(self, itr, eps):
        """Train the algorithm once.
        Args:
            itr (int): Iteration number.
            eps (EpisodeBatch): A batch of collected paths.
        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.
        """
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in eps.padded_rewards
            ]))
        valids = eps.lengths
        with torch.no_grad():
            baselines = self._value_function(obs, seq_idx=self.seq_idx)
        
        self._episode_reward_mean.append(rewards.mean().item())

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs, self.seq_idx)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat, advs_flat, self.seq_idx)


        self._old_policy.load_state_dict(self.policy.state_dict())

        # 매 step 마다 evaluate 할 필요가 없을듯?
        undiscounted_returns = 0
        # with torch.no_grad():
        policy_loss = self._compute_loss_with_adv(
            obs_flat, actions_flat, rewards_flat, advs_flat, self.seq_idx)
        vf_loss = self._value_function.compute_loss(
            obs_flat, returns_flat, seq_idx=self.seq_idx)
        kl = self._compute_kl_constraint(obs, self.seq_idx)
        policy_entropy = self._compute_policy_entropy(obs, self.seq_idx)

        end_time = time()


        if self._no_stats == False:
            policy_zero_cnt = sum(self.policy._stats['dormant'][-1000:]) / 1000
            value_zero_cnt = sum(self._value_function._stats['dormant'][-1000:]) / 1000

            policy_zero_cnt = sum(self.policy._stats['dormant'][-1000:]) / 1000
            value_zero_cnt = sum(self._value_function._stats['dormant'][-1000:]) / 1000

            epsilon = 1e-5

            value_last_weight = self._value_function.module._mean_module._output_layers[0][0].weight
            policy_last_weight = self.policy._module._mean_module._output_layers[self.seq_idx][0].weight

            value_hessian = weight_hessian(vf_loss, value_last_weight)
            policy_hessian = weight_hessian(policy_loss, policy_last_weight)

            value_hessian_rank = feature_rank(value_hessian, 1e-5)
            policy_hessian_rank = feature_rank(policy_hessian, 1e-5)

            print("policy / value hessian rank: ", policy_hessian_rank, value_hessian_rank)

            n = obs_flat.shape[0]

            value_normalized_feature = self._value_function._feature / np.sqrt(n)
            policy_normalized_feature = self.policy._feature.flatten(0, 1) / np.sqrt(n)

            value_feature_rank = feature_rank(value_normalized_feature, 1e-3)
            policy_feature_rank = feature_rank(policy_normalized_feature, 1e-5)

            print("policy / value feature rank: ", policy_feature_rank, value_feature_rank)

            vf_state_dict = copy.deepcopy(self._value_function.state_dict())
            policy_state_dict = copy.deepcopy(self.policy.state_dict())

        
            value_dev = weight_deviation(vf_state_dict, self.recent_vf_state_dict)
            policy_dev = weight_deviation(policy_state_dict, self.recent_policy_state_dict)

            print("policy / value weight deviation: ", policy_dev.item(), value_dev.item())

            self.recent_vf_state_dict = vf_state_dict
            self.recent_policy_state_dict = policy_state_dict

        if self._use_wandb:
            if self._no_stats == False:
                wandb.log({
                    'Running avg. of episode return': sum(self._episode_reward_mean) / len(self._episode_reward_mean),
                    'Policy loss': policy_loss.item(),
                    'Value loss': (vf_loss).item(),
                    'KL': (kl).item(),
                    'Speed (it/s)' : ((itr+1) / (end_time - self.start_time)),
                    'Policy dormant ratio': policy_zero_cnt,
                    'Value dormant ratio': value_zero_cnt,
                    'Value feature rank': value_feature_rank,
                    'Policy hessian rank': policy_hessian_rank,
                    'Value hessian rank': value_hessian_rank,
                    'Policy weight change': policy_dev.item(),
                    'Value weight change': value_dev.item(),
                })
            else:
                wandb.log({
                    'Running avg. of episode return': sum(self._episode_reward_mean) / len(self._episode_reward_mean),
                    'Policy loss': policy_loss.item(),
                    'Value loss': (vf_loss).item(),
                    'KL': (kl).item(),
                    'Speed (it/s)' : ((itr+1) / (end_time - self.start_time)),
                })

        self.results['Running avg. of episode return'].append(sum(self._episode_reward_mean) / len(self._episode_reward_mean))
        self.results['Policy loss'].append(policy_loss.item())
        self.results['Value loss'].append((vf_loss).item())
        self.results['KL'].append((kl).item())
        self.results['Speed (it/s)'].append(((itr+1)*2000 / (end_time - self.start_time)))

        if self._no_stats == False:
            self.results['Policy dormant ratio'].append(policy_zero_cnt)
            self.results['Value dormant ratio'].append(value_zero_cnt)
            self.results['Policy feature rank'].append(policy_feature_rank)
            self.results['Value feature rank'].append(value_feature_rank)
            self.results['Policy hessian rank'].append(policy_hessian_rank)
            self.results['Value hessian rank'].append(value_hessian_rank)
            self.results['Policy weight change'].append(policy_dev.item())
            self.results['Value weight change'].append(value_dev.item())

        print('STEP: {} '.format(itr),'policy loss: {:.6f} '.format(policy_loss.item()), 'Value loss: {:.6f} '.format((vf_loss).item()), 'Reward avg.: {:.6f}'.format(sum(self._episode_reward_mean) / len(self._episode_reward_mean)), 'Speed: {:.1f} it/s'.format((itr+1)*2000 / (end_time - self.start_time)))
            
        if self.seq_idx != getattr(self._sampler._envs[0], "cur_seq_idx"):
            print('Task change')
            print('Current task number =',self.seq_idx)
            # NOTE: Must call self.task_change before changeing self.seq_idx
            self.task_change(self.seq_idx)
            self.seq_idx = getattr(self._sampler._envs[0], "cur_seq_idx")
            print('Next task number =',self.seq_idx)
        
        
        return np.mean(self._episode_reward_mean)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.
        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.
        Returns:
            float: The average return in last epoch cycle.
        """
        last_return = None
        self.recent_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.recent_vf_state_dict = copy.deepcopy(self._value_function.state_dict())

        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr, seq_idx=self.seq_idx)
                last_return = self._train_once(trainer.step_itr, eps)

            self.save_results()
            trainer.step_itr += 1
            if trainer.step_itr % 10 == 0: self._evaluate_policy(trainer.step_itr)
        
        return last_return

    def _train(self, obs, actions, rewards, returns, advs, seq_idx):
        r"""Train the policy and value function with minibatch.
        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.
        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs):
            self._train_policy(*dataset, seq_idx=seq_idx)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset, seq_idx=seq_idx)

    def _infer_loss(self, pred_network, target_network):
        pred = pred_network.get_feature_prediction()
        target = target_network.get_feature_prediction().clone()
        return F.mse_loss(pred, self._infer_beta * target)
    
    def wasserstein_reg_loss(self, model, target):
        target.eval()
        loss = 0
        for p1, p2 in zip(model.parameters(), target.parameters()):
            sorted, _ = p1.flatten().sort()
            target_sorted, _ = p2.flatten().sort()
            loss += F.mse_loss(sorted, target_sorted)
        return self._wasserstein_lambda * loss

    def _train_policy(self, obs, actions, rewards, advantages, seq_idx):
        r"""Train the policy.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).
        """
        # pylint: disable=protected-access
        zero_optim_grads(self._policy_optimizer._optimizer)
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages, seq_idx)
        loss += self.cl_reg_loss(seq_idx)
        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_policy(obs, seq_idx)[0]
            loss += self._infer_loss(self.policy, self._infer_target_policy)
        if self._wasserstein:
            loss += self.wasserstein_reg_loss(self.policy, self._wasserstein_target_policy)
        loss.backward()
        self._policy_optimizer.step()

        return loss

    def _train_value_function(self, obs, returns, seq_idx = None):
        r"""Train the value function.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.
        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).
        """

        # pylint: disable=protected-access
        zero_optim_grads(self._vf_optimizer._optimizer)
        loss = self._value_function.compute_loss(obs, returns, seq_idx=seq_idx)
        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_vf.compute_loss(obs, returns, seq_idx=seq_idx)
            loss += self._infer_loss(self._value_function, self._infer_target_vf)
        if self._wasserstein:
            loss += self.wasserstein_reg_loss(self._value_function, self._wasserstein_target_vf)
        loss.backward()
        self._vf_optimizer.step()

        return loss

    def _compute_loss(self, obs, actions, rewards, valids, baselines, seq_idx):
        r"""Compute mean value of loss.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, P, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        advantages_flat = self._compute_advantage(rewards, valids, baselines)

        return self._compute_loss_with_adv(obs_flat, actions_flat,
                                           rewards_flat, advantages_flat, seq_idx)

    def _compute_loss_with_adv(self, obs, actions, rewards, advantages, seq_idx):
        r"""Compute mean value of loss.
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.
        """
        objectives = self._compute_objective(advantages, obs, actions, rewards, seq_idx)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs, seq_idx)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives.mean()

    def _compute_advantage(self, rewards, valids, baselines):
        r"""Compute mean value of loss.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.
        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.
        """
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self.max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def _compute_kl_constraint(self, obs, seq_idx):
        r"""Compute KL divergence.
        Compute the KL divergence between the old policy distribution and
        current policy distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).
        """
        with torch.no_grad():
            old_dist = self._old_policy(obs, seq_idx)[0]

        new_dist = self.policy(obs, seq_idx)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs, seq_idx):
        r"""Compute entropy value of probability distribution.
        Notes: P is the maximum episode length (self.max_episode_length)
        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.
        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy(obs, seq_idx)[0].entropy()
        else:
            policy_entropy = self.policy(obs, seq_idx)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, obs, actions, rewards, seq_idx):
        r"""Compute objective value.
        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.
        """
        del rewards
        log_likelihoods = self.policy(obs, seq_idx)[0].log_prob(actions)

        return log_likelihoods * advantages
    
    def _evaluate_policy(self, epoch):

        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_eps = []
        for seq_idx, eval_env in enumerate(self._eval_env):

            self.on_test_start(seq_idx)
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

    def ReDo(self, seq_idx):

        policy_network = self.policy._module._mean_module
        value_network = self._value_function.module._mean_module

        random_policy_network = self.random_policy._module._mean_module
        random_value_network = self._random_value_function.module._mean_module

        network_list = [policy_network, value_network]
        random_network_list = [random_policy_network, random_value_network]

        for network_idx, (random_network, network) in enumerate(zip(random_network_list, network_list)):
            random_layers = random_network._layers
            layers = network._layers
            pre_idx = -1
            for idx, (random_layer, layer) in enumerate(zip(random_layers, layers)):
                
                with torch.no_grad():
                    if pre_idx!=-1:
                        pre_zero_idx = network._stats['dormant_idx'][pre_idx]
                        temp = 1 - pre_zero_idx.float()
                        temp = temp.unsqueeze(0)
                        layer[0].weight.data *= temp

                    zero_idx = network._stats['dormant_idx'][idx]
                    mask = zero_idx.float().unsqueeze(-1)

                    layer[0].weight.data = (1-mask)*layer[0].weight.data + mask*random_layer[0].weight.data
                    mask = mask.squeeze()
                    layer[0].bias.data = (1-mask)*layer[0].bias.data + mask*random_layer[0].bias.data

                    pre_idx = idx
            
            zero_idx = network._stats['dormant_idx'][pre_idx]

            if network_idx == 0:
                next_seq_idx = seq_idx + 1
                network._output_layers[next_seq_idx][0].weight.data[:, zero_idx] = 0
            else:
                network._output_layers[0][0].weight.data[:, zero_idx] = 0
    
    def cl_reg_network(self):
        return [self.policy]
    def on_task_start(self, seq_idx):
        pass
    def on_test_start(self, seq_idx):
        pass
    def on_test_end(self, seq_idx):
        pass
    def cl_reg_loss(self, seq_idx):
        return 0
    def task_change(self, seq_idx):
        self.on_task_start(seq_idx)

        if self._ReDo and (seq_idx+1) < len(self._eval_env):
            self.ReDo(seq_idx)

        if self._policy_reset:
            self.policy.load_state_dict(self._random_policy_state_dict)
            self._old_policy.load_state_dict(self._random_policy_state_dict)
            print("Policy reset")
        if self._value_reset:
            self._value_function.load_state_dict(self._random_vf_state_dict)
            print("Value function reset")
    
    def save_models(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        #  model을 그냥 models에 저장한다
        if not os.path.exists('models/'):
            os.makedirs('models')
        if not os.path.exists('models/ppo_models'):
            os.makedirs('models/ppo_models')
        
        for net, name in zip(self.networks, self.networks_names):
            torch.save(net.state_dict(), './models/ppo_models/' + name + '_' + log_name + '.pt')

    def save_rollouts(self, log_name = None, buffer_size = int(1e6)):
        
        buffer = dict()
        seq_idx = 0
        eval_env = self._eval_env[0]
        
        episode_batch = obtain_evaluation_episodes(
                self.policy,
                eval_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
        buffer['observation'] = episode_batch.observations
        while buffer['observation'].shape[0] < buffer_size:
            episode_batch = obtain_evaluation_episodes(
                self.policy,
                eval_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
            observation = episode_batch.observations
            buffer['observation'] = np.concatenate((buffer['observation'], observation))
        buffer['observation'] = torch.Tensor(buffer['observation'][:buffer_size, :])
        buffer['observation'] = buffer['observation'].to(global_device())
        
        assert buffer['observation'].shape[0] == buffer_size

        if not os.path.exists('rollouts'):
            os.makedirs('rollouts')
        if not os.path.exists('rollouts/ppo_rollouts'):
            os.makedirs('rollouts/ppo_rollouts')


        path = './rollouts/ppo_rollouts/' + log_name + '.pkl'
        with open(path, "wb") as file:
            pickle.dump(buffer, file)

    def save_results(self, log_name = None):

        if log_name is None:
            log_name = self._log_name
        
        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        path = './logs/' + log_name + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._old_policy, self._value_function
        ]
    
    @property
    def networks_names(self):
        return [
            'policy', 'old_policy', 'vf'
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if self._infer:
            self._infer_target_policy.to(device)
            self._infer_target_vf.to(device)
        if self._wasserstein:
            self._wasserstein_target_policy.to(device)
            self._wasserstein_target_vf.to(device)
        if self._ReDo:
            self.random_policy.to(device)
            self._random_value_function.to(device)
