import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sys

from src.utils.environments import *
from src.utils.dqn import DQNAgent, DQNAgent_solo
import src.utils.plotting_utils  as plotting_utils

def main():
    # Number of physical cores on machine
    num_cpus = psutil.cpu_count(logical=False)

    # Start ray
    ray.init(logging_level="ERROR")

    num_agents = num_cpus if (sys.argv[2] == "max") else int(sys.argv[2])
    N_EPISODES = int(sys.argv[1])

    print("Num agents: ", num_agents)
    print("Num episodes: " , N_EPISODES)

    # Initialize workers
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), 1) for _ in range(num_agents)]

    # Setup config
    [agent.set_scheduler.remote((0, N_EPISODES-50), (0.2, 0.00)) for agent in agents]

    # Train 
    train_ids = [agent.train.remote(N_EPISODES, diffusion=False) for agent in agents]

    rewards = ray.get(train_ids)

    plotting_utils.plot_workers(rewards)
    plt.savefig("results/plots/workers.png")
    plotting_utils.plot_workers_aggregate(rewards)
    plt.savefig("results/plots/workers_agg.png")

    plt.show()

    print("Done!")

if __name__ == "__main__":
    main()