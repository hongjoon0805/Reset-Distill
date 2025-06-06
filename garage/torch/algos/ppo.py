"""Proximal Policy Optimization (PPO)."""
import torch

from garage.torch.algos import VPG
from garage.torch.optimizers import OptimizerWrapper

class PPO(VPG):
    """Proximal Policy Optimization (PPO).
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
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
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

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 seed = 0,
                 eval_env=None,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 lr_clip_range=2e-1,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 log_name = None,
                 num_evaluation_episodes=10,
                 q_reset=False,
                 policy_reset=False,
                 first_task = None,
                 use_wandb=True,
                 crelu=False,
                 infer=False,
                 wasserstein=0, 
                 ReDo=False,
                 no_stats=False,
                 multi_input=False,
                 policy_lr=5e-4,
                 value_lr=5e-4,
                 max_optimization_epochs=32,
                 minibatch_size=128):
        
        if policy_optimizer is None:
            policy_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=policy_lr)),
                policy,
                max_optimization_epochs=max_optimization_epochs,
                minibatch_size=minibatch_size)
        if vf_optimizer is None:
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=value_lr)),
                value_function,
                max_optimization_epochs=max_optimization_epochs,
                minibatch_size=minibatch_size)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         seed=seed,
                         eval_env=eval_env,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         log_name=log_name,
                         num_evaluation_episodes=num_evaluation_episodes,
                         q_reset=q_reset,
                         policy_reset=policy_reset,
                         first_task=first_task,
                         use_wandb=use_wandb,
                         crelu=crelu,
                         infer=infer,
                         wasserstein=wasserstein, 
                         ReDo=ReDo,
                         no_stats=no_stats, 
                         multi_input=multi_input)

        self._lr_clip_range = lr_clip_range

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
        # Compute constraint

        old_ll, new_ll = self._compute_nll(obs, actions, seq_idx)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)
    
    def _compute_nll(self, obs, actions, seq_idx):
        with torch.no_grad():
            old_ll = self._old_policy(obs, seq_idx)[0].log_prob(actions)
        new_ll = self.policy(obs, seq_idx)[0].log_prob(actions)

        return old_ll, new_ll
