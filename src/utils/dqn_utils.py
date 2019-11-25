from collections import namedtuple
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

class EpsilonScheduler(object):
    def __init__(self, t_range, eps_range):
        self.eps_start = eps_range[0]
        self.eps_end = eps_range[1]
        self.eps_range = eps_range[1] - eps_range[0]
        self.t_start = t_range[0]
        self.t_end = t_range[1]
        self.t_duration = self.t_end - self.t_start

    def __call__(self, t):
        if t < self.t_start:
            return self.eps_start
        if t > self.t_end:
            return self.eps_end
        return self.eps_start + \
            ((t-self.t_start)/self.t_duration) * (self.eps_range)

class MLP_DQN(nn.Module):
    def __init__(self, input_dim, output_dim, n_units=24):
        super(MLP_DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, output_dim)
        )

    def forward(self, x):
        return self.model(x.float())

def train(agents, num_episodes):
    promises = []
    for ep_id in range(num_episodes):
        for agent in agents:
            promises.append(agent.run_episode.remote(ep_id))
            agent.diffuse.remote(ep_id)
    
    return promises