from garage.replay_buffer import PathBuffer
import numpy as np
import torch
from garage import StepType, TimeStepBatch

class ExpertBuffer:

    def __init__(self, capacity, per_task):
        self._capacity_per_task=capacity
        self._per_task = per_task
        self._capacity=0
            
        if per_task:
            self.observation_buffer = []
            self.target_mean_buffer = []
            self.target_log_std_buffer = []
            self.task_idx_buffer = []
        else:
            self.observation_buffer = None
            self.target_mean_buffer = None
            self.target_log_std_buffer = None
            self.task_idx_buffer = None
    
    def add_observation_batch(self, observations, policy, seq_idx):
        """Add a EpisodeBatch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.

        """
        assert len(observations) == self._capacity_per_task
        
        BATCH_SIZE = 64
        
        means = []
        log_stds = []

        for i in range(0,len(observations),BATCH_SIZE):
            start = i
            if i+BATCH_SIZE > len(observations):
                end = len(observations)
            else:
                end = i+BATCH_SIZE
            with torch.no_grad():
                action_info = policy(observations[start:end], seq_idx)[1]
                mean, log_std = action_info['mean'], action_info['log_std']
                means.append(mean)
                log_stds.append(log_std)
        
        mean_targets = torch.cat(means)
        log_std_targets = torch.cat(log_stds)

        if self._per_task:
            self.observation_buffer.append(observations)
            self.target_mean_buffer.append(mean_targets)
            self.target_log_std_buffer.append(log_std_targets)
        else:
            if self.observation_buffer is None:
                self.observation_buffer = observations
                self.target_mean_buffer = mean_targets
                self.target_log_std_buffer = log_std_targets
                self.task_idx_buffer = torch.tensor([seq_idx]*self._capacity_per_task)
            else:
                self.observation_buffer=torch.cat([self.observation_buffer, observations])
                self.target_mean_buffer = torch.cat([self.target_mean_buffer, mean_targets])
                self.target_log_std_buffer = torch.cat([self.target_log_std_buffer, log_std_targets])
                self.task_idx_buffer = torch.cat([self.task_idx_buffer, torch.tensor([seq_idx]*self._capacity_per_task)])

        self._capacity += self._capacity_per_task
    
    def sample_expert_batch(self, batch_size, task_idx=None):
        if self._per_task:
            num_task = len(self.observation_buffer)
            idx = np.random.randint(self._capacity // num_task, size=batch_size // num_task)

            obs = self.observation_buffer[task_idx][idx]
            target_means = self.target_mean_buffer[task_idx][idx]
            target_log_stds = self.target_log_std_buffer[task_idx][idx]

            return obs, target_means, target_log_stds

        else:
            idx = np.random.randint(self._capacity, size=batch_size)
            obs = self.observation_buffer[idx]
            target_means = self.target_mean_buffer[idx]
            target_log_stds = self.target_log_std_buffer[idx]
            task_idxs = self.task_idx_buffer[idx]
            
            return obs, target_means, target_log_stds, task_idxs
    
    