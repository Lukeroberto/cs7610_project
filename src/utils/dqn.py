import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import random

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
    
#     def reset(self):
#         self.memory = []
#         self.position = 0

#     def __len__(self):
#         return len(self.memory)

# class EpsilonScheduler(object):
# 	def __init__(self, t_range, eps_range):
# 		self.eps_start = eps_range[0]
# 		self.eps_end = eps_range[1]
# 		self.eps_range = eps_range[1] - eps_range[0]
# 		self.t_start = t_range[0] 
# 		self.t_end = t_range[1]
# 		self.t_duration = self.t_end - self.t_start

# 	def __call__(self, t):
# 		if t < self.t_start: return self.eps_start
# 		if t > self.t_end: return self.eps_end
# 		return self.eps_start + \
# 			((t-self.t_start)/self.t_duration) * (self.eps_range)

class MLP_DQN(nn.Module):
	def __init__(self, input_dim, output_dim, n_units=32):
		super(MLP_DQN, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_dim, n_units),
			nn.BatchNorm1d(n_units),
			nn.ReLU(True),
			nn.Linear(n_units, n_units),
			nn.BatchNorm1d(n_units),
			nn.ReLU(True),
			nn.Linear(n_units, output_dim)
		)

	def forward(self, x):
		return self.model(x.float())

class DQNAgent():
	def __init__(self, env, epsilon):
		self.EPS = epsilon
		self.GAMMA = 0.98
		self.EP_LENGTH = 500
		self.step_counter = 0
		self.state = None

		self.env = env
		self.env.reset()
		self.to_torch = self.env.torch_state

		self.learning_rate = 1e-4
		self.reset_model()


	def reset_model(self):
		# resets model parameters
		self.model = MLP_DQN(self.env.state_dim, self.env.nA)
		self.optimizer = optim.Adam(self.model.parameters(),
									lr = self.learning_rate)

	def get_action(self, t_state, epsilon):
		if npr.random() > epsilon:
			self.model.eval()
			with torch.no_grad():
				return self.model.forward(t_state).argmax().view(1,1)
		else:
			return torch.tensor(npr.randint(self.env.nA), 
								dtype=torch.long).view(1,1)

	def optimize(self, n_steps):
		"""
		takes n_steps in the environment and then perform a batch 
		gradient update with the gathered experiences
		"""	
		state_batch = torch.zeros(n_steps, self.env.state_dim)
		action_batch = torch.zeros(n_steps, self.env.nA)
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
				self.state = self.env.reset()[0]
				self.step_counter = 0
		#endfor

		state_action_values = self.model.forward(state_batch) \
										.gather(1, action_batch)

		next_state_values = self.target.forward(next_state_batch) \
										.max(1)[0].detach()
		next_state_values[done_batch==1.] = 0.

		expected_state_action_values = (next_state_values * self.GAMMA) + \
										reward_batch

		loss = F.mse_loss(state_action_values, expected_state_action_values)
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.model.parameters():
			param.grad.data.clamp_(-1,1)
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
				dict_params[name1].data.copy_(0.5*param1.data + 0.5*dict_params[name1].data)

		self.model.load_state_dict(dict_params)






