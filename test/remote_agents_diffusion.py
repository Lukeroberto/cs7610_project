
import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sys

from src.utils.environments import ContinuousGridWorld
from src.utils.dqn import DQNAgent, DQNAgent_solo
import src.utils.plotting_utils  as plotting_utils
from src.utils.graph_utils import *

def main():
    # Number of physical cores on machine
    num_cpus = psutil.cpu_count(logical=False)

    # Start ray
    ray.init(logging_level="ERROR")

    N_EPISODES = int(sys.argv[1])
    num_agents = num_cpus if (sys.argv[2] == "max") else int(sys.argv[2])
    print(f"Num agents: {num_agents}")
    print(f"Num episodes: {N_EPISODES}")

    # Initialize workers
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), i, "1a") for i in range(num_agents)]

    # Set neighbors
    neighbors = generate_neighbor_graph(chain_adj(num_agents), agents)
    # print("Agent neighbors: ", neighbors)
    for agent in agents:
        agent.set_neighbors.remote(neighbors[agent])

    # Setup config
    [agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.0)) for agent in agents]

    # Train 
    print("Training...")
    train_ids = [agent.train.remote(N_EPISODES, diffusion=True) for agent in agents]

    # Save weights
    # [agent.save_weights.remote() for agent in agents]
    rewards = ray.get(train_ids)


    plotting_utils.plot_workers(rewards)
    plt.savefig("results/plots/workers_diffusion.png")
    plotting_utils.plot_workers_aggregate(rewards)
    plt.savefig("results/plots/workers_agg_diffusion.png")

    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()