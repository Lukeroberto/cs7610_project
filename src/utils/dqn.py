import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import ray

from src.utils.plotting_utils import *

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, input_dim, output_dim, n_units=64):
        super(MLP_DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, output_dim)
        )

    def forward(self, x):
        return self.model(x.float())

class DQNAgent():
    def __init__(self, p_id, env, epsilon):
        self.p_id = p_id
        self.EPS = epsilon
        self.GAMMA = 0.98
        self.EP_LENGTH = 200
        self.step_counter = 0

        self.env = env
        self.state = self.env.reset()
        self.to_torch = self.env.torch_state

        self.learning_rate = 1e-4
        self.reset_model()

    def save_weights(self):
        torch.save(self.model.state_dict(), "results/{}.pth".format(self.p_id))

    def save_returns(self):
        # current time, num_opt, num_diff, returns
        pass

    def reset_model(self):
        # resets model parameters
        self.model = MLP_DQN(self.env.state_dim, self.env.nA)
        self.target = MLP_DQN(self.env.state_dim, self.env.nA)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

    def get_action(self, t_state, epsilon):
        if npr.random() > epsilon:
            self.model.eval()
            with torch.no_grad():
                return self.model.forward(t_state).argmax().view(1, 1)
        else:
            return torch.tensor(npr.randint(self.env.nA),
                                dtype=torch.long).view(1, 1)

    def optimize(self, n_steps):
        """
        takes n_steps in the environment and then perform a batch 
        gradient update with the gathered experiences
        """
        state_batch = torch.zeros(n_steps, self.env.state_dim)
        action_batch = torch.zeros(n_steps, 1, dtype=torch.long)
        next_state_batch = torch.zeros(n_steps, self.env.state_dim)
        reward_batch = torch.zeros(n_steps)
        done_batch = np.zeros(n_steps)

        for step_id in range(n_steps):
            t_state = self.to_torch(self.state)

            t_action = self.get_action(t_state, self.EPS)
            self.state, reward, done, _ = self.env.step(t_action.item())
            t_next_state = self.to_torch(self.state)

            state_batch[step_id] = t_state
            action_batch[step_id] = t_action
            next_state_batch[step_id] = t_next_state
            reward_batch[step_id] = reward
            done_batch[step_id] = done

            self.step_counter += 1
            if done or self.step_counter == self.EP_LENGTH:
                self.state = self.env.reset()
                self.step_counter = 0
        # endfor

        state_action_values = self.model.forward(state_batch) \
            .gather(1, action_batch).squeeze()

        next_state_values = self.target.forward(next_state_batch) \
            .max(1)[0].detach()
        next_state_values[done_batch == 1.] = 0.

        expected_state_action_values = (next_state_values * self.GAMMA) + \
            reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def diffuse(self, nbr_model):
        """
        Given model of neighbor that is same architecture as this agent,
        update model params to be average of the two model params
        """
        params1 = self.model.named_parameters()
        params2 = nbr_model.named_parameters()

        dict_params = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params:
                dict_params[name1].data.copy_(
                    0.5*param1.data + 0.5*dict_params[name1].data)

        self.model.load_state_dict(dict_params)

@ray.remote
class DQNAgent_solo():
    def __init__(self, env, id, logging=True):
        self.p_id = id 

        self.GAMMA = 0.98
        self.EP_LENGTH = 50

        self.memory_size = 50000
        self.target_update_interval = 30
        self.memory = ReplayMemory(self.memory_size)
        self.scheduler = EpsilonScheduler((0, 1e10), (1.0, 1.0))
        self.batch_size = 64
        self.step_counter = 0
        self.opt_freq = 1

        self.env = env
        self.state = self.env.reset()
        self.to_torch = self.env.torch_state

        self.learning_rate = 1e-4
        self.reset_model()

        self.logging = logging

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def get_model(self):
        return self.model

    def save_weights(self):
        torch.save(self.model.state_dict(), "results/models/agent{}.pth".format(self.p_id))

    def reset_model(self):
        # resets model parameters
        self.model = MLP_DQN(self.env.state_dim, self.env.nA)
        self.target = MLP_DQN(self.env.state_dim, self.env.nA)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        self.memory.reset()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def get_action(self, t_state, epsilon):
        if npr.random() > epsilon:
            self.model.eval()
            with torch.no_grad():
                return self.model.forward(t_state).argmax().view(1, 1)
        else:
            return torch.tensor(npr.randint(self.env.nA),
                                dtype=torch.long).view(1, 1)

    def set_scheduler(self, t_limits, eps_limits):
        self.scheduler = EpsilonScheduler(t_limits, eps_limits)

    def add_sample(self, t_state, t_action, t_next_state, t_reward):
        self.memory.push(t_state, t_action, t_next_state, t_reward)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = self.model.forward(state_batch) \
                                        .gather(1, action_batch) 

        next_state_values = self.target.forward(next_state_batch) \
                        .detach().max(1)[0].unsqueeze(1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, ep_id):
        EPS = self.scheduler(ep_id)

        state = self.env.reset()

        for step_id in range(self.EP_LENGTH):
            t_state = self.to_torch(state)
            t_action = self.get_action(t_state, EPS)
            next_state, reward, done, _ = self.env.step(t_action.item())

            t_next_state = self.to_torch(next_state)
            t_reward = torch.tensor(reward, dtype=torch.float).view(1, 1)

            self.add_sample(t_state, t_action, t_next_state, t_reward)
            state = np.copy(next_state)

            self.step_counter += 1
            if self.step_counter % self.opt_freq == 0:
                self.optimize()
                self.step_counter = 0

            if done:
                break
        return done
        
        # endfor
    def train(self, num_episodes, diffusion=False):
        returns = np.zeros(num_episodes)
        for ep_id in range(num_episodes):
            returns[ep_id] = self.run_episode(ep_id)

            if self.logging and ep_id > 100 and ep_id % 50 == 0:
                plt.figure()
                plt.plot(smooth(returns[:ep_id], 100))

                results_dir = f"results/agent_{self.p_id}" 
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                plt.savefig(results_dir + "/smoothed_returns.png")
                plt.close()

            if diffusion:
                self.diffuse()

        return returns

    def diffuse(self):
        beta = 0.5 #The interpolation parameter    

        # Get weights for neighbors
        for n in self.neighbors:

            # block to get neighbor weights
            neighbor_model = ray.wait([n.get_model.remote()], timeout=0.1)

            if (len(neighbor_model[0]) == 0):
                return 

            for w in neighbor_model[0]:
                neighbor_model = ray.get(w)

            # Get named parameter dicts
            params1 = neighbor_model.named_parameters()
            params2 = self.model.named_parameters()

            # Average weights
            dict_params2 = dict(params2)
            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)

            # Set my parameters to average
            self.model.load_state_dict(dict_params2)
