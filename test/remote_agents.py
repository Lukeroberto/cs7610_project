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
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), i) for i in range(num_agents)]

    # Setup config
<<<<<<< Updated upstream
    [agent.set_scheduler.remote((0, N_EPISODES-50), (1.0, 0.01)) for agent in agents]
=======
    [agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.01)) for agent in agents]
>>>>>>> Stashed changes

    # Train 
    train_ids = [agent.train.remote(N_EPISODES, diffusion=False) for agent in agents]

    rewards = ray.get(train_ids)

<<<<<<< Updated upstream
    plotting_utils.plot_workers(rewards, smoothing=200)
    plt.savefig("results/aggregate/workers.png")
    plotting_utils.plot_workers_aggregate(rewards, smoothing=200)
    plt.savefig("results/aggregate/workers_agg.png")
=======
    plotting_utils.plot_workers(rewards)
    # plt.savefig("results/plots/workers.png")
    plotting_utils.plot_workers_aggregate(rewards)
    # plt.savefig("results/plots/workers_agg.png")
>>>>>>> Stashed changes

    plt.show()

    print("Done!")

if __name__ == "__main__":
    main()