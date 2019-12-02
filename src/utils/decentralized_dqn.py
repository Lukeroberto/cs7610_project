import numpy as np
import numpy.random as npr
import torch
import os
import matplotlib.pyplot as plt
import random
import ray

from src.utils.plotting_utils import *
from src.utils.dqn_utils import *

@ray.remote
class DQNAgent():
    def __init__(self, env, ids, opt=True, logging=True):
        # Several IDs used in testing
        self.p_id = ids[0]
        self.test_id = ids[1]

        # Hyperparameters
        self.GAMMA = 0.98
        self.EP_LENGTH = 50
        self.learning_rate = 1e-4
        self.memory_size = 100000
        self.target_update_interval = 30
        self.batch_size = 64
        self.step_counter = 0
        self.opt_freq = 1

        # Replay buffer and schedule
        self.memory = ReplayMemory(self.memory_size)
        self.scheduler = EpsilonScheduler((0, 1e10), (1.0, 1.0))

        # Save env and get an initial state
        self.env = env
        self.to_torch = self.env.torch_state

        # Reset model
        self.reset_model()

        # Logging and diffusion related structures
        self.logging = logging
        self.offline = []
        self.opt = opt

        self.temp_model = None
        self.temp_target = None

        self.returns =  []
        self.train_diffusions = []

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
    
    def get_id(self):
        return self.p_id

    def get_returns(self):
        return np.array(self.returns)
    
    def get_avg_returns(self):
        if len(self.returns) == 0:
            return 0

        return np.array(self.returns)[-50:].mean()
    
    def get_diffusion_counts(self):
        return np.array(self.train_diffusions)

    def get_model(self):
        return dict(self.model.named_parameters())

    def get_model_and_avg_returns(self):
        return self.get_model(), self.get_avg_returns()

    def save_weights(self):
        torch.save(self.model.state_dict(), "results/example_agent.pth".format(self.p_id))

    def reset_model(self):
        # resets model parameters
        self.model = MLP_DQN(self.env.state_dim, self.env.nA)
        self.target = MLP_DQN(self.env.state_dim, self.env.nA)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        self.memory.reset()
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def load_model(self, model):
        self.model.load_state_dict(model.state_dict())
    
    def load_torch_file(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def load_target(self, model):
        self.target.load_state_dict(model.state_dict())

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

    def add_sample(self, t_state, t_action, t_next_state, t_reward, t_done):
        self.memory.push(t_state, t_action, t_next_state, t_reward, t_done)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.model.forward(state_batch) \
                                        .gather(1, action_batch) 

        next_state_values = self.target.forward(next_state_batch) \
                        .detach().max(1)[0].unsqueeze(1)
        next_state_values[done_batch] = 0.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_episode(self):
        state = self.env.reset()
        self.model.eval()
        for step_id in range(self.EP_LENGTH):
            t_state = self.to_torch(state)
            t_action = self.model.forward(t_state).argmax().view(1, 1)
            next_state, reward, done, _ = self.env.step(t_action.item())
            state = np.copy(next_state)
            if done:
                break

        self.returns.append(done)
        return done

    def run_episode(self, ep_id):
        EPS = self.scheduler(ep_id)

        state = self.env.reset()
        for step_id in range(self.EP_LENGTH):
            t_state = self.to_torch(state)
            t_action = self.get_action(t_state, EPS)
            next_state, reward, done, _ = self.env.step(t_action.item())

            t_next_state = self.to_torch(next_state)
            t_reward = torch.tensor(reward, dtype=torch.float).view(1, 1)
            t_done = torch.tensor(done, dtype=torch.long).view(1,1)
            self.add_sample(t_state, t_action, t_next_state, t_reward, t_done)
            state = np.copy(next_state)

            if done:
                break
            
        for _ in range(step_id+1):
            if self.opt:
                self.optimize()
        
        if self.test_id == "fail_stop" and \
                self.p_id == 0 and \
                ep_id > 500 and \
                ep_id < 1200:

            self.reset_model()
            self.update_target()
            done = -1
        
        if ep_id % self.target_update_interval == 0:
            self.update_target()

        self.returns.append(done)

        # Log periodically
        if self.logging and ep_id > 100 and ep_id % 15 == 0:
            rewards = np.array(self.returns)
            diffusions = np.array(self.train_diffusions).cumsum()

            plot_training_progress(rewards, diffusions, [self.p_id, self.test_id, ep_id])

        return done

    def diffuse(self, ep_id):
        beta = 0.5 #The interpolation parameter    

        # Get weights for neighbors
        if self.test_id == "fail_stop" and self.p_id != 0 and ep_id > 500 and ep_id < 1200:
            temp_neighbors = self.neighbors[1:]
        elif self.test_id == "fail_stop" and self.p_id == 0 and ep_id > 500 and ep_id < 1200:
            temp_neighbors = []
        elif self.test_id == "network_partition" and self.p_id in [2, 3] and ep_id > 500 and ep_id < 1200:
            if self.p_id == 2:
                temp_neighbors = self.neighbors[:-1]
            if self.p_id == 3:
                temp_neighbors = self.neighbors[1:]
        else:
            temp_neighbors = self.neighbors
        

        neighbor_models = [n.get_model_and_avg_returns.remote() for n in temp_neighbors]

        ready, not_ready = ray.wait(neighbor_models, 
                                    num_returns=len(neighbor_models),
                                    timeout=0.1)
        num_diffusions = len(ready)
        if (num_diffusions == 0):
            self.train_diffusions.append(0)
            return 

        for w in ready:
            neighbor_model, avg_return = ray.get(w)

            # Dont diffuse if your avg returns are worse than mine
            # if "3" in self.test_id and avg_return < self.get_avg_returns():
            #     continue

            # Get named parameter dicts
            params1 = neighbor_model
            dict_params2 = self.get_model()

            # Average weights
            for name1, param1 in params1.items():
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)

            # Set my parameters to average
            self.model.load_state_dict(dict_params2)
        
        self.train_diffusions.append(num_diffusions)
