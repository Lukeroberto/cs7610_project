import numpy as np 
import matplotlib.pyplot as plt 
import datetime
import ray
import pickle

from src.utils.environments import ContinuousGridWorld
from src.utils.decentralized_dqn import DQNAgent
from src.utils.centralized_dqn import DQNAgent_central, CentralizedRunner
from src.utils.graph_utils import *
from src.utils.dqn_utils import train

def main(N_EPISODES=2000, N_AGENTS=6, load=True):
    if load:
         with open("results/topology_comparison/data.pickle", "rb") as handle:
            DATA = pickle.load(handle)
    else:
        ray.init(logging_level="ERROR")
        topologies = [chain_adj, full_adj, spoke_adj]
        topologies_names = ["chain", "connected", "spoke"]
        DATA = dict()

        for i, topology in enumerate(topologies):
            mat = topology(N_AGENTS)
            agents = [DQNAgent.remote(ContinuousGridWorld(), [i, ''], logging=False) for i in range(N_AGENTS)]
            neighbors = generate_neighbor_graph(mat, agents)
            for agent in agents:
                agent.set_neighbors.remote(neighbors[agent])
            
            train(agents, N_EPISODES)

            DATA[topologies_names[i]] = ray.get([agent.get_returns.remote() for agent in agents])

        with open("results/topology_comparison/data.pickle", "wb") as handle:
                pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)

    W = 100
    plt.figure()
    for k,v in DATA.items():
        v = np.array(v).mean(axis=0)
        plt.plot(np.convolve(v, np.ones(W)/W, mode="valid"))

    plt.legend(DATA.keys())
    plt.show()

if __name__ == "__main__":
    main()