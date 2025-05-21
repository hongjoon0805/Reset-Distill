"""A replay buffer that efficiently stores and can sample whole paths."""
import collections
import pickle
import torch

import numpy as np
import random

from garage import StepType, TimeStepBatch
from garage.replay_buffer import PathBuffer
from garage.replay_buffer import SumSegmentTree, MinSegmentTree

# expert buffer가 PathBuffer를 상속해서 짜면 편할듯

# 사실상 priority만 제대로 저장하면 될듯.
# Pathbuffer에 일단 transition 모두 저장하고, sample할 때 priority 기반으로 sample하면 된다.
# Pathbuffer 상속하고 sample batch, compute_weight, priority update 정도만 짜두면 될 것 같음.

class PrioritizedReplayBuffer(PathBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        capacity_in_transitions,
        env_spec = None, 
        alpha = 0.5
    ):
        """Initialization."""
        assert alpha >= 0
        
        super().__init__(capacity_in_transitions, env_spec)
        self._max_priority, self._tree_ptr = 1.0, 0
        self._alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self._capacity:
            tree_capacity *= 2

        self._sum_tree = SumSegmentTree(tree_capacity)
        self._min_tree = MinSegmentTree(tree_capacity)
    
    def add_episode_batch(self, episodes):
        super().add_episode_batch(episodes)

        for eps in episodes.split():
            eps_len = len(eps.observations)
            for _ in range(eps_len):
                self._sum_tree[self._tree_ptr] = self._max_priority ** self._alpha
                self._min_tree[self._tree_ptr] = self._max_priority ** self._alpha
                self._tree_ptr = (self._tree_ptr + 1) % self._capacity

    def sample_timesteps(self, batch_size, beta=0.6):
        assert beta>0

        indices = self._sample_proportional(batch_size)
        timesteps = super().sample_timesteps(batch_size, idx=indices)
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return timesteps, weights, indices

        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self._transitions_stored

            self._sum_tree[idx] = priority ** self._alpha
            self._min_tree[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
            
    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        indices = []
        p_total = self._sum_tree.sum(0, self._transitions_stored - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self._sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self._min_tree.min() / self._sum_tree.sum()
        max_weight = (p_min * self._transitions_stored) ** (-beta)
        
        # calculate weights
        p_sample = self._sum_tree[idx] / self._sum_tree.sum()
        weight = (p_sample * self._transitions_stored) ** (-beta)
        weight = weight / max_weight
        
        return weight
