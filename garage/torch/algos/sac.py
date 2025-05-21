"""This modules creates a sac model in PyTorch."""
from collections import deque
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

# yapf: disable
from garage import log_performance, obtain_evaluation_episodes, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device, state_dict_to, np_to_torch
from garage.torch._functions import list_to_tensor, zero_optim_grads, weight_deviation, weight_hessian, feature_rank
from time import time

import wandb
import os
import pickle

from time import time

# yapf: enable

def load_model(model, model_name, first_task, seed):
    # Load policy
    copied_model = copy.deepcopy(model)
    name = model_name.format(first_task, seed)
    loaded_state_dict = torch.load('./models/sac_models/'+name, map_location=global_device())

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


class SAC(RLAlgorithm):
    """A SAC Model in Torch.

    Based on Soft Actor-Critic and Applications:
        https://arxiv.org/abs/1812.05905

    Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr (int): Number of optimization steps that should
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.
        temporal_regularization_factor (float): coefficient that determines
            the temporal regularization penalty as defined in CAPS as lambda_t
        spatial_regularization_factor (float): coefficient that determines
            the spatial regularization penalty as defined in CAPS as lambda_s
        spatial_regularization_eps (float): sigma of the normal distribution
            from with spatial regularization observations are drawn,
            in caps this is defined as epsilon_s
    """

    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            seed=0,
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            use_exploration = False,
            q_reset = False,
            policy_reset = False,
            first_task = None,
            temporal_regularization_factor=0.,
            spatial_regularization_factor=0.,
            spatial_regularization_eps=1.,
            log_name = None, 
            use_wandb=True,
            infer = False,
            crelu=False,
            wasserstein = 0, 
            ReDo = False, 
            no_stats=False, 
            multi_input=False):

        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env
        self._seed = seed
        self._infer = infer
        self._wasserstein = (wasserstein > 0)
        self._ReDo = ReDo
        self._no_stats = no_stats
        self._multi_input = multi_input

        # Total number of CL tasks
        self.masks = None

        self._log_name = log_name
        self._use_wandb = use_wandb

        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        if isinstance(env_spec,list):
            max_episode_length = env_spec[0].max_episode_length
        else:
            max_episode_length = env_spec.max_episode_length
        self.max_episode_length = max_episode_length
        self._max_episode_length_eval = max_episode_length
        self._beta = 0.05
        self._rho = 0.00001

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation
        self._use_exploration = use_exploration
        self._q_reset = q_reset
        self._policy_reset = policy_reset
        self._first_task = first_task

        self._temporal_regularization_factor = temporal_regularization_factor
        self._spatial_regularization_factor = spatial_regularization_factor
        self._spatial_regularization_dist = torch.distributions.Normal(
            0, spatial_regularization_eps)

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler

        self._reward_scale = reward_scale
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._policy_lr)
        self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
                                              lr=self._qf_lr)
        self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
                                              lr=self._qf_lr)
        
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                if isinstance(self.env_spec, list):
                    self._target_entropy = [-np.prod(spec.action_space.shape).item() for spec in self.env_spec]
                    
                else:
                    self._target_entropy = -np.prod(
                            self.env_spec.action_space.shape).item()

        self._reset_alpha()
        
        self.episode_rewards = deque(maxlen=30)
        self.recent_trajectory = RecentTrajectory(maxlen=10000)

        self.global_step = 0
        # self.global_step = -self._gradient_steps
        self.seq_idx = 0
        self.start_time = time()
        self.begin = self.start_time
        self.results = {}

        self.results['Running avg. of episode return'] = []
        self.results['Policy loss'] = []
        self.results['Q loss'] = []
        self.results['Alpha'] = []
        self.results['Speed (it/s)'] = []

        if self._no_stats == False:
        
            self.results['Policy zero ratio'] = []
            self.results['Qf1 zero ratio'] = []
            self.results['Qf2 zero ratio'] = []
            self.results['Policy feature rank'] = []
            self.results['Qf1 feature rank'] = []
            self.results['Qf2 feature rank'] = []
            self.results['Policy hessian rank'] = []
            self.results['Qf1 hessian rank'] = []
            self.results['Qf2 hessian rank'] = []
            self.results['Policy weight change'] = []
            self.results['Qf1 weight change'] = []
            self.results['Qf2 weight change'] = []
        

        self._random_qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        self._random_qf2_state_dict = copy.deepcopy(self._qf2.state_dict())
        self._random_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        
        # For ReDo
        self._random_qf1 = copy.deepcopy(self._qf1)
        self._random_qf2 = copy.deepcopy(self._qf2)
        self.random_policy = copy.deepcopy(self.policy)

        random_policy_network = self.random_policy._module._shared_mean_log_std_network
        

        if infer:
            self._infer_target_policy = copy.deepcopy(self.policy)
            self._infer_target_qf1 = copy.deepcopy(self._qf1)
            self._infer_target_qf2 = copy.deepcopy(self._qf2)
            self._infer_alpha = 1.
            self._infer_beta = 10.
            print("Use InFeR Loss")

        if wasserstein:
            self._wasserstein_target_policy = copy.deepcopy(self.policy)
            self._wasserstein_target_qf1 = copy.deepcopy(self._qf1)
            self._wasserstein_target_qf2 = copy.deepcopy(self._qf2)
            self._wasserstein_lambda = wasserstein
            print("Use Wasserstein Regularization")

        if self._q_reset:
            print('Reset Q when task is changed')
        
        if self._policy_reset:
            print('Reset policy when task is changed')

        if self._first_task is not None:
            if 'DMC' in self._first_task:
                policy_model_name = 'policy_dm_control_sac_{}_1000000_{}.pt'
                qf1_model_name = 'qf1_dm_control_sac_{}_1000000_{}.pt'
                target_qf1_model_name = 'target_qf1_dm_control_sac_{}_1000000_{}.pt'
                qf2_model_name = 'qf2_dm_control_sac_{}_1000000_{}.pt'
                target_qf2_model_name = 'target_qf2_dm_control_sac_{}_1000000_{}.pt'

                
            else:
                policy_model_name = 'policy_metaworld_sac_{}_3000000_{}.pt'
                qf1_model_name = 'qf1_metaworld_sac_{}_3000000_{}.pt'
                target_qf1_model_name = 'target_qf1_metaworld_sac_{}_3000000_{}.pt'
                qf2_model_name = 'qf2_metaworld_sac_{}_3000000_{}.pt'
                target_qf2_model_name = 'target_qf1_metaworld_sac_{}_3000000_{}.pt'
                if crelu:
                    policy_model_name = 'policy_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    qf1_model_name = 'qf1_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    target_qf1_model_name = 'target_qf1_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    qf2_model_name = 'qf2_CReLU_metaworld_sac_{}_3000000_{}.pt'
                    target_qf2_model_name = 'target_qf2_CReLU_metaworld_sac_{}_3000000_{}.pt'
                if wasserstein:
                    policy_model_name = 'policy_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    qf1_model_name = 'qf1_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    target_qf1_model_name = 'target_qf1_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    qf2_model_name = 'qf2_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
                    target_qf2_model_name = 'target_qf2_Wasserstein_0.1_metaworld_sac_{}_3000000_{}.pt'
            
                
                
            load_model(self.policy, policy_model_name, self._first_task, self._seed)
            load_model(self._qf1, qf1_model_name, self._first_task, self._seed)
            load_model(self._target_qf1, target_qf1_model_name, self._first_task, self._seed)
            load_model(self._qf2, qf2_model_name, self._first_task, self._seed)
            load_model(self._target_qf2, target_qf2_model_name, self._first_task, self._seed)

            
                

            if self._q_reset:
                print('############################################################')
                print('                        Q-reset!!!!!                        ')
                print('############################################################')
                self._qf1.load_state_dict(self._random_qf1_state_dict)
                self._qf2.load_state_dict(self._random_qf2_state_dict)

                self._target_qf1 = copy.deepcopy(self._qf1)
                self._target_qf2 = copy.deepcopy(self._qf2)
            
            if self._policy_reset:
                print('############################################################')
                print('                     Policy-reset!!!!!                      ')
                print('############################################################')
                self.policy.load_state_dict(self._random_policy_state_dict)



    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        self.recent_policy_state_dict = copy.deepcopy(self.policy.state_dict())
        self.recent_qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        self.recent_qf2_state_dict = copy.deepcopy(self._qf2.state_dict())
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for env in self._sampler._envs:
            env.reset()
            
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                
                if self._use_exploration and batch_size is not None:
                    if self._multi_input:
                        discount_tensor = torch.full((1000,), self._discount)
                        filter = torch.cumprod(discount_tensor, dim=0) / self._discount
                    else:
                        discount_tensor = torch.full((500,), self._discount)
                        filter = torch.cumprod(discount_tensor, dim=0) / self._discount
                    
                    episodes_list = []
                    return_list = []
                    for seq in range(self.seq_idx+1):
                        episodes = trainer.obtain_samples(
                            trainer.step_itr, seq, batch_size)
                        episodes_list.append(episodes)
                        ret = 0
                        for path in episodes:
                            rewards = path['rewards']
                            ret = np.sum(rewards * filter.numpy())
                        return_list.append(ret.item())
                        
                    best_ret_arg = np.argsort(np.array(return_list))[-1]
                    
                    trainer.step_episode = episodes_list[best_ret_arg]

                else:
                    trainer.step_episode = trainer.obtain_samples(
                            trainer.step_itr, self.seq_idx, batch_size)
                
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(path['rewards'])
                    self.recent_trajectory.append(path)
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))

                

                for _ in range(self._gradient_steps):
                    
                    policy_loss, qf1_loss, qf2_loss = self.train_once(self.seq_idx)
                    self.global_step += 1
                    with torch.no_grad():
                        alpha = self._log_alpha.exp()
                    end_time = time()
                    
                    if self.global_step % 1000 == 0:

                        if self._no_stats == False:

                            # Dormant neurons
                            policy_zero_cnt = sum(self.policy._stats['zero_ratio'][-1000:]) / 1000
                            qf1_zero_cnt = sum(self._qf1._stats['zero_ratio'][-1000:]) / 1000
                            qf2_zero_cnt = sum(self._qf2._stats['zero_ratio'][-1000:]) / 1000


                            # Feature rank, Hessian
                            eps = 0.001 # eps = 0.01 in Lyle et al. (2022); tried but the feature rank of policy is too small(below 10)

                            recent_obs = self.recent_trajectory.observation
                            recent_samples = self.recent_trajectory.samples

                            qf1_loss_hess, qf2_loss_hess = self._critic_objective(recent_samples, self.seq_idx)

                            action_dists, new_actions, log_pi_new_actions = self._get_policy_output(recent_obs, self.seq_idx)

                            policy_loss_hess = self._actor_objective(recent_samples, new_actions,
                                                                log_pi_new_actions, seq_idx=self.seq_dx)
                            policy_loss_hess += self._caps_regularization_objective(
                                action_dists, recent_samples, self.seq_idx)
                            
                            qf1_last_weight = self._qf1._output_layers[0][0].weight
                            qf2_last_weight = self._qf2._output_layers[0][0].weight
                            policy_last_weight = self.policy._module._shared_mean_log_std_network._output_layers[2*self.seq_idx][0].weight
                            
                            qf1_hessian = weight_hessian(qf1_loss_hess, qf1_last_weight)
                            qf2_hessian = weight_hessian(qf2_loss_hess, qf2_last_weight)
                            policy_hessian = weight_hessian(policy_loss_hess, policy_last_weight)

                            qf1_hessian_rank = feature_rank(qf1_hessian, 1e-5)
                            qf2_hessian_rank = feature_rank(qf2_hessian, 1e-5)
                            policy_hessian_rank = feature_rank(policy_hessian, 1e-5)

                            print("policy / qf1 / qf2 hessian rank: ", policy_hessian_rank, qf1_hessian_rank, qf2_hessian_rank)


                            policy_normalized_feature =  self.policy._feature / np.sqrt(self.recent_trajectory.maxlen)
                            qf1_normalized_feature =  self._qf1._feature / np.sqrt(self.recent_trajectory.maxlen)
                            qf2_normalized_feature =  self._qf2._feature / np.sqrt(self.recent_trajectory.maxlen)                        

                            policy_feature_rank = feature_rank(policy_normalized_feature, eps)
                            qf1_feature_rank = feature_rank(qf1_normalized_feature, eps)
                            qf2_feature_rank = feature_rank(qf2_normalized_feature, eps)

                            print("policy / qf1 / qf2 feature rank: ", policy_feature_rank, qf1_feature_rank, qf2_feature_rank)

                            # Weight deviation
                            policy_state_dict = copy.deepcopy(self.policy.state_dict())
                            qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
                            qf2_state_dict = copy.deepcopy(self._qf2.state_dict())

                            policy_dev = weight_deviation(policy_state_dict, self.recent_policy_state_dict)
                            qf1_dev = weight_deviation(qf1_state_dict, self.recent_qf1_state_dict)
                            qf2_dev = weight_deviation(qf2_state_dict, self.recent_qf2_state_dict)

                            print("Policy / qf1 / qf2 weight deviation: ", policy_dev.item(), qf1_dev.item(), qf2_dev.item())

                            self.recent_policy_state_dict = policy_state_dict
                            self.recent_qf1_state_dict = qf1_state_dict
                            self.recent_qf2_state_dict = qf2_state_dict

                        if self._use_wandb:
                            if self._no_stats == False:
                                wandb.log({
                                    'Running avg. of episode return': sum(self.episode_rewards) / len(self.episode_rewards),
                                    'Policy loss': policy_loss.item(),
                                    'Q loss': (qf1_loss + qf2_loss).item(),
                                    'Alpha': alpha.item(),
                                    'Speed (it/s)' : (self.global_step / (end_time - self.start_time)),
                                    'Policy zero ratio': policy_zero_cnt,
                                    'Qf1 zero ratio': qf1_zero_cnt,
                                    'Qf2 zero ratio': qf2_zero_cnt,
                                    'Policy feature rank': policy_feature_rank,
                                    'Qf1 feature rank': qf1_feature_rank,
                                    'Qf2 feature rank': qf2_feature_rank,
                                    'Policy hessian rank': policy_hessian_rank,
                                    'Qf1 hessian rank': qf1_hessian_rank,
                                    'Qf2 hessian rank': qf2_hessian_rank,
                                    'Policy weight change': policy_dev.item(),
                                    'Qf1 weight change': qf1_dev.item(),
                                    'Qf2 weight change': qf2_dev.item(),  
                                })
                            else:
                                wandb.log({
                                    'Running avg. of episode return': sum(self.episode_rewards) / len(self.episode_rewards),
                                    'Policy loss': policy_loss.item(),
                                    'Q loss': (qf1_loss + qf2_loss).item(),
                                    'Alpha': alpha.item(),
                                    'Speed (it/s)' : (self.global_step / (end_time - self.start_time))
                                })


                        self.results['Running avg. of episode return'].append(sum(self.episode_rewards) / len(self.episode_rewards))
                        self.results['Policy loss'].append(policy_loss.item())
                        self.results['Q loss'].append((qf1_loss + qf2_loss).item())
                        self.results['Alpha'].append(alpha.item())
                        self.results['Speed (it/s)'].append((self.global_step / (end_time - self.start_time)))

                        if self._no_stats == False:
                            self.results['Policy zero ratio'].append(policy_zero_cnt)
                            self.results['Qf1 zero ratio'].append(qf1_zero_cnt)
                            self.results['Qf2 zero ratio'].append(qf2_zero_cnt)
                            self.results['Policy feature rank'].append(policy_feature_rank)
                            self.results['Qf1 feature rank'].append(qf1_feature_rank)
                            self.results['Qf2 feature rank'].append(qf2_feature_rank)
                            self.results['Policy hessian rank'].append(policy_hessian_rank)
                            self.results['Qf1 hessian rank'].append(qf1_hessian_rank)
                            self.results['Qf2 hessian rank'].append(qf2_feature_rank)
                            self.results['Policy weight change'].append(policy_dev.item())
                            self.results['Qf1 weight change'].append(qf1_dev.item())
                            self.results['Qf2 weight change'].append(qf2_dev.item())
                        

                        print('STEP: {} '.format(self.global_step),'policy loss: {:.2f} '.format(policy_loss.item()), 'Q loss: {:.6f} '.format((qf1_loss + qf2_loss).item()), 'Alpha: {:.7f}'.format(alpha.item()), 'Reward avg.: {:.7f}'.format(sum(self.episode_rewards) / len(self.episode_rewards)), 'Speed: {:.1f} it/s'.format(self.global_step / (end_time - self.start_time)))
                
                if self.seq_idx != getattr(self._sampler._envs[0], "cur_seq_idx"):
                    print('Task change')
                    print('Current task number =',self.seq_idx)
                    # NOTE: Must call self.task_change before changing self.seq_idx
                    self.task_change(self.seq_idx)
                    self.seq_idx = getattr(self._sampler._envs[0], "cur_seq_idx")
                    print('Next task number =',self.seq_idx)
            
            last_return = self._evaluate_policy(trainer.step_itr)
            self.save_results()
            trainer.step_itr += 1

        return np.mean(last_return)

    def train_once(self, seq_idx, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)
            samples = as_torch_dict(samples)

            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples, seq_idx)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha

        """
        del samples_data
        log_alpha = self._log_alpha
        return log_alpha

    def _temperature_objective(self, log_pi, samples_data, seq_idx=None):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            if isinstance(self.env_spec, list):
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy[seq_idx])).mean()
            else:
                alpha_loss = (-(self._get_log_alpha(samples_data)) *
                            (log_pi.detach() + self._target_entropy)).mean()
        return alpha_loss

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

    def ReDo(self, seq_idx):

        # Layer만 생각할게 아니라, output layer도 생각해야한다.
        policy_network = self.policy._module._shared_mean_log_std_network

        random_policy_network = self.random_policy._module._shared_mean_log_std_network

        network_list = [policy_network, self._qf1, self._qf2]
        random_network_list = [random_policy_network, self._random_qf1, self._random_qf2]

        for network_idx, (random_network, network) in enumerate(zip(random_network_list, network_list)):
            random_layers = random_network._layers
            layers = network._layers
            pre_idx = -1
            for idx, (random_layer, layer) in enumerate(zip(random_layers, layers)):
                
                with torch.no_grad():
                    if pre_idx!=-1:
                        pre_zero_idx = network._stats['zero_idx'][pre_idx]
                        temp = 1 - pre_zero_idx.float()
                        temp = temp.unsqueeze(0)
                        layer[0].weight.data *= temp


                    zero_idx = network._stats['zero_idx'][idx]
                    mask = zero_idx.float().unsqueeze(-1)

                    layer[0].weight.data = (1-mask)*layer[0].weight.data + mask*random_layer[0].weight.data
                    mask = mask.squeeze()
                    layer[0].bias.data = (1-mask)*layer[0].bias.data + mask*random_layer[0].bias.data

                    pre_idx = idx
            
            
            zero_idx = network._stats['zero_idx'][pre_idx]

            if network_idx == 0:
                next_seq_idx = seq_idx + 1
                network._output_layers[2*next_seq_idx][0].weight.data[:, zero_idx] = 0
                network._output_layers[2*next_seq_idx+1][0].weight.data[:, zero_idx] = 0
            else:
                network._output_layers[0][0].weight.data[:, zero_idx] = 0


    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions, seq_idx=None):
        """Compute the Policy/Actor loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(obs, new_actions, seq_idx),
                                      self._qf2(obs, new_actions, seq_idx))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        
        return policy_objective

    def _critic_objective(self, samples_data, seq_idx):
        """Compute the Q-function/critic loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1(obs, actions, seq_idx=seq_idx)
        q2_pred = self._qf2(obs, actions, seq_idx=seq_idx)

        new_next_actions_dist = self.policy(next_obs, seq_idx)[0]
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        
        qf1 = self._target_qf1(next_obs, new_next_actions, seq_idx=seq_idx)
        qf2 = self._target_qf2(next_obs, new_next_actions, seq_idx=seq_idx)
        
        target_q_values = torch.min(qf1,qf2).flatten() - (alpha * new_log_pi)
        
        
        with torch.no_grad():
            q_target = rewards * self._reward_scale + (
                1. - terminals) * self._discount * target_q_values

        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def _caps_regularization_objective(self, action_dists, samples_data, seq_idx):
        """Compute the spatial and temporal regularization loss as in CAPS.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            action_dists (torch.distribution.Distribution): Distributions
                returned from the policy after feeding through observations.

        Returns:
            torch.Tensor: combined regularization loss
        """
        # torch.tensor is callable and the recommended way to create a scalar
        # tensor
        # pylint: disable=not-callable

        if self._temporal_regularization_factor:
            next_action_dists = self.policy(
                samples_data['next_observation'], seq_idx)[0]
            temporal_loss = self._temporal_regularization_factor * torch.mean(
                torch.cdist(action_dists.mean, next_action_dists.mean, p=2))
        else:
            temporal_loss = torch.tensor(0.)

        if self._spatial_regularization_factor:
            obs = samples_data['observation']
            noisy_action_dists = self.policy(
                obs + self._spatial_regularization_dist.sample(obs.shape), seq_idx)[0]
            spatial_loss = self._spatial_regularization_factor * torch.mean(
                torch.cdist(action_dists.mean, noisy_action_dists.mean, p=2))
        else:
            spatial_loss = torch.tensor(0.)

        return temporal_loss + spatial_loss

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

    def optimize_policy(self, samples_data, seq_idx):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data, seq_idx)

        if self._infer:
            obs = samples_data['observation']
            actions = samples_data['action']
            with torch.no_grad():
                _ = self._infer_target_qf1(obs, actions, seq_idx=seq_idx)
                _ = self._infer_target_qf2(obs, actions, seq_idx=seq_idx)
            qf1_loss += self._infer_alpha * self._infer_loss(self._qf1, self._infer_target_qf1)
            qf2_loss += self._infer_alpha * self._infer_loss(self._qf2, self._infer_target_qf2)

        if self._wasserstein:
            qf1_loss += self.wasserstein_reg_loss(self._qf1, self._wasserstein_target_qf1)
            qf2_loss += self.wasserstein_reg_loss(self._qf2, self._wasserstein_target_qf2)

        zero_optim_grads(self._qf1_optimizer)
        qf1_loss.backward()
        self._qf1_optimizer.step()

        zero_optim_grads(self._qf2_optimizer)
        qf2_loss.backward()
        self._qf2_optimizer.step()


        # action_dists = self.policy(obs, seq_idx)[0]
        # new_actions_pre_tanh, new_actions = (
        #     action_dists.rsample_with_pre_tanh_value())
        # log_pi_new_actions = action_dists.log_prob(
        #     value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        action_dists, new_actions, log_pi_new_actions = self._get_policy_output(obs, seq_idx)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions, seq_idx=seq_idx)
        policy_loss += self._caps_regularization_objective(
            action_dists, samples_data, seq_idx)
        policy_loss += self.cl_reg_loss(seq_idx)
        

        if self._infer:
            with torch.no_grad():
                _ = self._infer_target_policy(obs, seq_idx)[0]
            policy_loss += self._infer_alpha * self._infer_loss(self.policy, self._infer_target_policy)

        if self._wasserstein:
            policy_loss += self.wasserstein_reg_loss(self.policy, self._wasserstein_target_policy)

        zero_optim_grads(self._policy_optimizer)
        policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data, seq_idx=seq_idx)
            zero_optim_grads(self._alpha_optimizer)
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _get_policy_output(self, obs, seq_idx):

        action_dists = self.policy(obs, seq_idx)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        
        return action_dists, new_actions, log_pi_new_actions

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
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            seq_idx = -1, # dummy
            max_episode_length = self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount,
                                      results=self.results, 
                                      use_wandb=self._use_wandb)
        return last_return

    def _reset_alpha(self):
        
        if self._use_automatic_entropy_tuning:
            self._log_alpha = list_to_tensor([self._initial_log_entropy
                                              ]).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                              lr=self._policy_lr)
        else:
            self._log_alpha = list_to_tensor([self._fixed_alpha]).log()

    def task_change(self, seq_idx):
        self.on_task_start(seq_idx)

        if self._ReDo and (seq_idx+1) < len(self._eval_env):
            self.ReDo(seq_idx)

        self.replay_buffer.clear()
        self._reset_alpha()
        self.recent_trajectory.clear()

        if self._q_reset:
            self._qf1.load_state_dict(self._random_qf1_state_dict)
            self._qf2.load_state_dict(self._random_qf2_state_dict)

        qf1_state_dict = copy.deepcopy(self._qf1.state_dict())
        qf2_state_dict = copy.deepcopy(self._qf2.state_dict())

        self._target_qf1.load_state_dict(qf1_state_dict)
        self._target_qf2.load_state_dict(qf2_state_dict)
    
    def save_models(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        #  Save models into 'models/sac_models'
        if not os.path.exists('models/'):
            os.makedirs('models')
        if not os.path.exists('models/sac_models'):
            os.makedirs('models/sac_models')

        for net, name in zip(self.networks, self.networks_names):
            torch.save(net.state_dict(), './models/sac_models' + name + '_' + log_name + '.pt')

        
    def save_buffers(self, log_name = None):

        if log_name is None:
            log_name = self._log_name

        # Save buffers into 'buffers/sac_buffers'
        if not os.path.exists('buffers'):
            os.makedirs('buffers')
        if not os.path.exists('buffers/sac_buffers'):
            os.makedirs('buffers/sac_buffers')

        buffer_data = self.replay_buffer.get_all_transitions()
        buffer_data = as_torch_dict(buffer_data)
        path = './buffers/sac_buffers' + log_name + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(buffer_data, f)
    
    def save_rollouts(self, log_name = None, buffer_size = int(1e6)):
        
        buffer = dict()
        seq_idx = 0
        eval_env = self._eval_env[0]
        observations = []
        obs_len = 0

        while obs_len < buffer_size:
            episode_batch = obtain_evaluation_episodes(
                self.policy,
                eval_env,
                seq_idx,
                self._max_episode_length_eval,
                num_eps=self._num_evaluation_episodes,
                deterministic=self._use_deterministic_evaluation)
            observation = episode_batch.observations
            observations.append(observation)
            obs_len += len(observation)
        
        buffer['observation'] = np.concatenate((observations))
        buffer['observation'] = torch.Tensor(buffer['observation'][:buffer_size, :])
        buffer['observation'] = buffer['observation'].to(global_device())
        
        assert buffer['observation'].shape[0] == buffer_size

        if not os.path.exists('rollouts'):
            os.makedirs('rollouts')
        if not os.path.exists('rollouts/sac_rollouts'):
            os.makedirs('rollouts/sac_rollouts')
        
        path = './rollouts/sac_rollouts/rollouts_' + log_name + '.pkl'
        with open(path, "wb") as file:
            pickle.dump(buffer, file)

    
    def save_results(self, log_name = None):

        if log_name is None:
            log_name = self._log_name
        
        path = './logs/' + log_name + '.pkl'

        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        with open(path, 'wb') as f:
            pickle.dump(self.results, f)

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2
        ]
    @property
    def networks_names(self):

        return [
            'policy', 'qf1', 'qf2', 'target_qf1', 'target_qf2'
        ]

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
            self._infer_target_qf1.to(device)
            self._infer_target_qf2.to(device)

        if self._wasserstein:
            self._wasserstein_target_policy.to(device)
            self._wasserstein_target_qf1.to(device)
            self._wasserstein_target_qf2.to(device)
        
        if self._ReDo:
            self.random_policy.to(device)
            self._random_qf1.to(device)
            self._random_qf2.to(device)

        if not self._use_automatic_entropy_tuning:
            self._log_alpha = list_to_tensor([self._fixed_alpha
                                              ]).log().to(device)
        else:
            self._log_alpha = self._log_alpha.detach().to(
                device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)
            self._alpha_optimizer.load_state_dict(
                state_dict_to(self._alpha_optimizer.state_dict(), device))
            self._qf1_optimizer.load_state_dict(
                state_dict_to(self._qf1_optimizer.state_dict(), device))
            self._qf2_optimizer.load_state_dict(
                state_dict_to(self._qf2_optimizer.state_dict(), device))
            self._policy_optimizer.load_state_dict(
                state_dict_to(self._policy_optimizer.state_dict(), device))


class RecentTrajectory:
    def __init__(self, maxlen = 10000):
        self._observation = []
        self._action = []
        self._reward = []
        self._next_observation = []
        self._terminal = []

        self.maxlen = maxlen

    def _concat(self, x, y):
        if x is None:
            return y
        else:
            return np.concatenate([x, y])[-self.maxlen:]

    def append(self, path):
        obs, action = path['observations'], path['actions']
        reward = path['rewards']
        next_obs = path['next_observations']
        terminal = np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
        
        

        self._observation.append(obs)
        self._action.append(action)
        self._reward.append(reward)
        self._next_observation.append(next_obs)
        self._terminal.append(terminal)
    
    @property
    def observation(self):
        observation = np.concatenate(self._observation)[-self.maxlen:]
        return np_to_torch(observation)
    
    @property
    def action(self):
        action = np.concatenate(self._action)[-self.maxlen:]
        return np_to_torch(action)
    
    @property
    def reward(self):
        reward = np.concatenate(self._reward)[-self.maxlen:]
        return np_to_torch(reward)

    @property
    def next_observation(self):
        next_observation = np.concatenate(self._next_observation)[-self.maxlen:]
        return np_to_torch(next_observation)
    
    @property
    def terminal(self):
        terminal = np.concatenate(self._terminal)[-self.maxlen:]
        return np_to_torch(terminal)
    
    @property
    def samples(self):
        dic = dict(
            observation = self.observation,
            action = self.action,
            reward = self.reward,
            terminal = self.terminal,
            next_observation = self.next_observation
        )
        return dic
    
    def clear(self):
        self._observation = []
        self._action = []
        self._reward = []
        self._next_observation = []
        self._terminal = []
