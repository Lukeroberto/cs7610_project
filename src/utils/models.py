import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP_DQN(nn.Module):
	def __init__(self, input_dim, output_dim, n_units=32):
		super(MLP_DQN, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_dim, n_units),
			nn.BatchNorm1d(n_units),
			)

	def forward(self, x):
		return self.model(x.float())