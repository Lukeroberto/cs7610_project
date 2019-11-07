import numpy as np 
import numpy.random as npr
import torch
import gym

class RandomizedCartpole():
	def __init__(self, rseed=None):
		self.env = gym.make("CartPole-v1").env
		self.nA = 2
		self.state_dim = 4
		if rseed is not None:
			np.random.seed(rseed)

		# randomly perturb environment params
		self.env.masscart += npr.uniform(-0.1, 0.1)
		self.env.total_mass = self.env.masscart+self.env.masspole
		self.env.length += npr.uniform(-0.05, 0.05)
		self.polemass_length = self.env.masspole * self.env.length
				
	def step(self, action):
		return self.env.step(action)

	def reset(self):
		return self.env.reset()

	def torch_state(self, obs):
		return torch.tensor(obs).view(-1, len(obs))