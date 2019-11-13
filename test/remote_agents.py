
import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from src.utils.environments import RandomizedCartpole
from src.utils.dqn import DQNAgent, DQNAgent_solo
import src.utils.plotting_utils  as plotting_utils

# Number of physical cores on machine
num_cpus = psutil.cpu_count(logical=False)
print(f"Num cpus: {num_cpus}")

# Start ray
ray.init(logging_level="ERROR")

N_EPISODES = 250

# Initialize workers
agents = [DQNAgent_solo.remote(RandomizedCartpole(), 1) for _ in range(num_cpus)]

# Setup config
[agent.set_scheduler.remote((0, N_EPISODES-50), (0.2, 0.00)) for agent in agents]

# Train 
train_ids = [agent.train.remote(N_EPISODES) for agent in agents]

rewards = ray.get(train_ids)

plotting_utils.plot_workers(rewards)
plotting_utils.plot_workers_aggregate(rewards)

plt.show()