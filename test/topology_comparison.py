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

def main(N_TRIALS=5, N_EPISODES=2000, N_AGENTS=30, load=False):
    if load:
         with open("results/topology_comparison/data.pickle", "rb") as handle:
            DATA = pickle.load(handle)
    else:
        ray.init(logging_level="ERROR")
        topologies = [chain_adj, full_adj, spoke_adj]
        topologies_names = ["chain", "connected", "spoke"]
        DATA = dict()

        for trial_id in range(N_TRIALS):
            print(f"starting trial {trial_id}")
            for i, topology in enumerate(topologies):
                mat = topology(N_AGENTS)
                agents = [DQNAgent.remote(ContinuousGridWorld(), [i, ''], logging=False) for i in range(N_AGENTS)]
                [agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.01)) for agent in agents]
                neighbors = generate_neighbor_graph(mat, agents)
                for agent in agents:
                    agent.set_neighbors.remote(neighbors[agent])
                
                train(agents, N_EPISODES)
                if topologies_names[i] not in DATA:
                    DATA[topologies_names[i]] = []
                DATA[topologies_names[i]].append(ray.get([agent.get_returns.remote() for agent in agents]))

        with open("results/topology_comparison/data.pickle", "wb") as handle:
            pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)

    W = 100
    plt.figure()
    for k,V in DATA.items():
        V = [np.convolve(np.array(v).mean(axis=0), np.ones(W)/W, mode="valid") for v in V]
        avg = np.array(V).mean(axis=0)
        std = np.array(V).std(axis=0)
        plt.plot(avg, label=k)
        plt.fill_between(np.arange(len(avg)), avg-std, avg+std, alpha=0.3)
        # p = plt.plot(np.convolve(v[0], np.ones(W)/W, mode="valid"), label=k)
        # for a in v[1:]:
        #     plt.plot(np.convolve(a, np.ones(W)/W, mode="valid"), color=p[0].get_color(),label='')
    plt.ylim((0,1))
    plt.legend()
    plt.savefig("results/topology_comparison/example.png")

if __name__ == "__main__":
    main()