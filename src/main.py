import argparse
import psutil

from src.utils.dqn import *
from src.utils.environments import *
from src.utils.graph_utils import *
from src.utils.plotting_utils import *
from src.utils.parser import *


# Number of physical cores on machine
NUM_CORES = psutil.cpu_count(logical=False)

# Start ray
ray.init(logging_level="ERROR")

def main():

    p = test_parser()
    args = p.parse_args()

    returns = test_dict[args.test](args)

    plotting_utils.plot_workers(rewards, smoothing=200)
    plt.savefig(f"results/test_{args.test}/workers.png")
    plotting_utils.plot_workers_aggregate(rewards, smoothing=200)
    plt.savefig(f"results/test_{args.test}/workers_agg.png")



def test_1a(args):
    NUM_EPISODES = int(args.length)

    # Get graph
    fully_connected = full_adj(NUM_CORES)

    # Initialize workers
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), i, args.test) for i in range(NUM_CORES)]

    # Setup config
    [agent.set_scheduler.remote((0, NUM_EPISODES-50), (1.0, 0.01)) for agent in agents]

    # Set neighbors
    neighbors = generate_neighbor_graph(fully_connected, agents)
    for agent in agents:
        agent.set_neighbors.remote(neighbors[agent])

    # Train 
    train_ids = [agent.train.remote(NUM_EPISODES) for agent in agents]

    return ray.get(train_ids)



def test_1b(args):
    NUM_EPISODES = int(args.length)

    # Get graph
    spoke = spoke_adj(NUM_CORES)

    # Initialize workers
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), i, args.test) for i in range(NUM_CORES)]

    # Setup config
    [agent.set_scheduler.remote((0, NUM_EPISODES-50), (1.0, 0.01)) for agent in agents]

    # Set neighbors
    neighbors = generate_neighbor_graph(spoke, agents)
    for agent in agents:
        agent.set_neighbors.remote(neighbors[agent])

    # Train 
    train_ids = [agent.train.remote(NUM_EPISODES) for agent in agents]

    return ray.get(train_ids)

def test_1c(args):
    NUM_EPISODES = int(args.length)

    # Get graph
    chain = chain_adj(NUM_CORES)

    # Initialize workers
    agents = [DQNAgent_solo.remote(ContinuousGridWorld(), i, args.test) for i in range(NUM_CORES)]

    # Setup config
    [agent.set_scheduler.remote((0, NUM_EPISODES-50), (1.0, 0.01)) for agent in agents]

    # Set neighbors
    neighbors = generate_neighbor_graph(chain, agents)
    for agent in agents:
        agent.set_neighbors.remote(neighbors[agent])

    # Train 
    train_ids = [agent.train.remote(NUM_EPISODES) for agent in agents]

    return ray.get(train_ids)

def test_2a(args):
    raise NotImplementedError

def test_2b(args):
    raise NotImplementedError

def test_3a(args):
    raise NotImplementedError

def test_3b(args):
    raise NotImplementedError

test_dict = {
    "1a": test_1a,
    "1b": test_1b,
    "1c": test_1c,
    "2a": test_2a,
    "2b": test_2b,
    "3a": test_3a,
    "3b": test_3b,
}

if __name__ == "__main__":
    main()