
import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from src.utils.environments import RandomizedCartpole
from src.utils.dqn import DQNAgent, DQNAgent_solo
import src.utils.plotting_utils  as plotting_utils
from src.utils.graph_utils import *

# Number of physical cores on machine
num_cpus = psutil.cpu_count(logical=False)

# Start ray
ray.init(logging_level="ERROR")

N_EPISODES = 10
num_agents = 2
print(f"Num agents: {num_agents}")

# Initialize workers
agents = [DQNAgent_solo.remote(RandomizedCartpole(), 1) for _ in range(num_agents)]

# Set neighbors
neighbors = generate_neighbor_graph(chain_adj(num_agents), agents)
print("Agent neighbors: ", neighbors)
for agent in agents:
    agent.set_neighbors.remote(neighbors[agent])

# Setup config
[agent.set_scheduler.remote((0, N_EPISODES-50), (0.2, 0.00)) for agent in agents]

# Train 
print("Training...")
train_ids = [agent.train.remote(N_EPISODES, diffusion=True) for agent in agents]

rewards = ray.get(train_ids)

plotting_utils.plot_workers(rewards)
plt.savefig("test/plots/workers.png")
plotting_utils.plot_workers_aggregate(rewards)
plt.savefig("test/plots/workers_agg.png")