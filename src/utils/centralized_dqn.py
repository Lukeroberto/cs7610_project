import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import random
import ray

from src.utils.dqn_utils import *

@ray.remote
class CentralizedRunner(object):
    # this will collect experiences and calculate the gradient
    def __init__(self, env, actor_id, gamma=0.98, lr=1e-4):
        self.env = env
        self.to_torch = env.torch_state
        self.id = actor_id
        self.model = MLP_DQN(self.env.state_dim, self.env.nA)
        self.target = MLP_DQN(self.env.state_dim, self.env.nA)
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr = self.lr)

    def calc_gradient(self, params, target_params, batch_size, eps=1.0):
        self.model.load_state_dict(params)
        self.target.load_state_dict(target_params)
        batch = self.get_batch(batch_size, eps)

        q_vals = self.model.forward(batch["S"]) \
                                        .gather(1, batch["A"])
        qp_vals = self.target.forward(batch["Sp"]) \
                        .detach().max(1)[0].unsqueeze(1)
        qp_vals[batch["D"]] = 0.
        target_vals = (qp_vals * self.gamma) + batch["R"]
        loss = F.mse_loss(q_vals, target_vals)
        
        self.optimizer.zero_grad()
        loss.backward()

        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.cpu().numpy())
            else:
                grads.append(None)
        return grads, batch["R"].numpy().max(), self.id

    def get_action(self, t_state, epsilon):
        if npr.random() > epsilon:
            self.model.eval()
            with torch.no_grad():
                return self.model.forward(t_state).argmax().view(1, 1)
        else:
            return torch.tensor(npr.randint(self.env.nA),
                                dtype=torch.long).view(1, 1)

    def get_experience(self, params, num_exp, eps=1.0):
        self.model.load_state_dict(params)
        batch = self.get_batch(num_exp, eps)
        return batch, batch["R"].numpy().max(), self.id

    def get_batch(self, num_steps, eps=1.0):
        S = []
        A = []
        Sp = []
        R = []
        D = []
        
        going = True
        while going:
            state = self.env.reset()
            for step_id in range(num_steps):
                t_state = self.to_torch(state)
                t_action = self.get_action(t_state, eps)
                next_state, reward, done, _ = self.env.step(t_action.item())
                t_next_state = self.to_torch(next_state)
                t_reward = torch.tensor(reward, dtype=torch.float).view(1, 1)
                t_done = torch.tensor(done, dtype=torch.long).view(1,1)
                state = np.copy(next_state)

                S.append(t_state)
                A.append(t_action)
                Sp.append(t_next_state)
                R.append(t_reward)
                D.append(t_done)
                if len(S) == num_steps: 
                    going = False
                    break
                if done: break

        return {"S": torch.cat(S), 
                "A": torch.cat(A),
                "Sp": torch.cat(Sp),
                "R" : torch.cat(R),
                "D" : torch.cat(D)}

class DQNAgent_central():
    def __init__(self, env, buffer_size=100000):
        self.GAMMA = 0.98
        self.EP_LENGTH = 50

        self.memory_size = buffer_size
        self.memory = ReplayMemory(self.memory_size)
        self.batch_size = 64

        self.env = env
        self.state = self.env.reset()
        self.to_torch = self.env.torch_state

        self.learning_rate = 1e-4
        self.reset_model()

    def reset_model(self):
        # resets model parameters
        self.model = MLP_DQN(self.env.state_dim, self.env.nA)
        self.target = MLP_DQN(self.env.state_dim, self.env.nA)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
        self.memory.reset()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def add_sample(self, t_state, t_action, t_next_state, t_reward, t_done):
        self.memory.push(t_state, t_action, t_next_state, t_reward, t_done)

    def run_episode(self):
        state = self.env.reset()
        self.model.eval()
        for step_id in range(self.EP_LENGTH):
            t_state = self.to_torch(state)
            t_action = self.model.forward(t_state).argmax().view(1, 1)
            next_state, reward, done, _ = self.env.step(t_action.item())
            state = np.copy(next_state)
            if done:
                break
        return done

    def add_batch(self, batch):
        S = batch["S"].unsqueeze(1)
        A = batch["A"].unsqueeze(1)
        Sp = batch["Sp"].unsqueeze(1)
        R = batch["R"].unsqueeze(1)
        D = batch["D"].unsqueeze(1)
        for i in range(batch["D"].shape[0]):
            self.add_sample(S[i],
                            A[i],
                            Sp[i],
                            R[i],
                            D[i])

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