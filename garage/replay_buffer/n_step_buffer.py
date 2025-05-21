"""A replay buffer that efficiently stores and can sample whole paths."""
import collections
import pickle
import torch

import numpy as np

from garage import StepType, TimeStepBatch
from garage.replay_buffer import PathBuffer

class NStepBuffer(PathBuffer):
    """N-step buffer


    Args:
        n_step : the number of steps for N-step learning

    """

    def __init__(self, capacity_in_transitions, n_step=1, env_spec=None,  discount=0.99):
        super().__init__(capacity_in_transitions, env_spec)
        assert n_step >= 1

        self._n_step = n_step
        self._discount = discount

    def add_episode_batch(self, episodes):
        """Add a EpisodeBatch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.

        """
        if self._env_spec is None:
            self._env_spec = episodes.env_spec
        env_spec = episodes.env_spec
        obs_space = env_spec.observation_space
        for eps in episodes.split():
            terminals = np.array([
                step_type == StepType.TERMINAL for step_type in eps.step_types
            ],
                                 dtype=bool)
            
            next_observations, rewards, terminals = self.get_n_step_info(eps.next_observations, eps.rewards, terminals, self._discount)

            path = {
                'observations': obs_space.flatten_n(eps.observations),
                'next_observations':
                obs_space.flatten_n(next_observations),
                'actions': env_spec.action_space.flatten_n(eps.actions),
                'rewards': rewards.reshape(-1, 1),
                'terminals': terminals.reshape(-1, 1),
            }
            self.add_path(path)


    def get_n_step_info(self, next_observations, rewards, terminals, discount):
        total_len = len(rewards)

        new_next_observations = np.zeros_like(next_observations)
        new_rewards = np.zeros_like(rewards)
        new_terminals = np.zeros_like(terminals)

        for i in range(total_len):
            n_step_idx = min([i+self._n_step-1, total_len-1])
            rew, next_obs, done = rewards[n_step_idx], next_observations[n_step_idx], terminals[n_step_idx]

            for n in reversed(range(self._n_step-1)):
                idx = min([i+n, total_len-1])
                r, n_o, d = rewards[idx], next_observations[idx], terminals[idx]

                rew = r + discount * rew * (1 - d)
                next_obs, done = (n_o, d) if d else (next_obs, done)

            new_next_observations[i] = next_obs
            new_rewards[i] = rew
            new_terminals[i] = done

        return new_next_observations, new_rewards, new_terminals


    