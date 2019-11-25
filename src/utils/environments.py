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
		# self.env.masscart += npr.uniform(-0.1, 0.1)
		# self.env.total_mass = self.env.masscart+self.env.masspole
		# self.env.length += npr.uniform(-0.05, 0.05)
		# self.polemass_length = self.env.masspole * self.env.length

				
	def step(self, action):
		return self.env.step(action)

	def reset(self):
		return self.env.reset()

	def torch_state(self, obs):
		return torch.tensor(obs).view(-1, len(obs))

class ContinuousGridWorld():
	def __init__(self, n=30, rseed=None):
		self.state = None
		self.state_dim = 2
		self.state_shape = (n,n)
		self.bounds = np.array(((0.,0.), (1.,1.)))
		self.n = n

		self.nA = 8
		self.dstep = 1.0/n
		angles = np.linspace(0, 2*np.pi,num=self.nA, endpoint=False)
		self._steps = self.dstep*np.array([(np.cos(a), np.sin(a)) for a in angles])

		self.goal = np.array((0.5,0.5))
		self.start = (0.2, 0.2)

		self.n_sigma = 0.00

	def random_state(self):
		return np.random.random(size = 2)

	def step(self, action):
		delta = self._steps[action]
		self.state = np.clip(self.state + delta + np.random.normal(scale=self.n_sigma,size=self.state_dim), 
							self.bounds[0], 
							self.bounds[1])
		reward = (np.linalg.norm(np.subtract(self.goal, self.state)) < 1.5*self.dstep).astype(int)
		done = True if reward==1 else False
		return self.state, reward, done, None

	def reset(self):
		self.state = np.array(self.random_state())
		return self.state

	def torch_state(self, state=None):
		if state is None:
			state = self.state
		return torch.tensor(state).view(-1, self.state_dim)