import psutil
from tqdm import tqdm 
import os

from src.utils.decentralized_dqn import *
from src.utils.environments import *
from src.utils.graph_utils import *
from src.utils.plotting_utils import *
from src.utils.parser import *

results_dir = f"results/test_network_partition/" 
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

def setup(args, graph):
    # Initialize workers
    agents = [DQNAgent.remote(ContinuousGridWorld(), [i, args.test]) for i in range(NUM_CORES)]

    # Setup config
    [agent.set_scheduler.remote((0, NUM_EPISODES-50), (0.6, 0.01)) for agent in agents]

    # Set neighbors
    neighbors = generate_neighbor_graph(graph, agents)
    for agent in agents:
        agent.set_neighbors.remote(neighbors[agent])

    return agents

def train(agents):
    rewards = np.zeros((len(agents), NUM_EPISODES))
    for ep_id in tqdm(range(NUM_EPISODES)):
        temp = []
        for agent_id, agent in enumerate(agents):
            temp.append(agent.run_episode.remote(ep_id))
            agent.diffuse.remote(ep_id)
        
        temp = [ray.get(t) for t in temp]
        rewards[:, ep_id] = temp

    return rewards

# Number of physical cores on machine
NUM_CORES = psutil.cpu_count(logical=False)

# Start ray
ray.init(logging_level="ERROR")

p = test_parser()
args = p.parse_args()

NUM_EPISODES = int(args.length)

# One of these architectures
graph = network_partition()

#Baseline
try:
    print("Loading baseline data")
    baseline_rewards = np.load("results/test_network_partition/baseline_rewards.npy")
except:
    print("Generating baseline data")
    args.test = "default_network_partition"
    baseline_agents = setup(args, graph)
    baseline_rewards = train(baseline_agents)
    np.save("results/test_network_partition/baseline_rewards.npy", baseline_rewards)

# Network Partition
try:
    test_rewards = np.load("results/test_network_partition/test_rewards.npy")
    print("Loading test data")
except:
    print("Generating test data")
    args.test = "network_partition"
    test_agents = setup(args, graph)
    test_rewards = train(test_agents)
    np.save("results/test_network_partition/test_rewards.npy", test_rewards)

# Plot Baseline
plt.figure()
plt.plot(smooth(baseline_rewards[0], 100), label="Baseline", color="C0")
[plt.plot(smooth(worker, 100), color="C0", alpha=0.2) for worker in baseline_rewards[1:]]

# Plot Partition 1
plt.plot(smooth(test_rewards[0], 100), label="Network Partition 1", color="C1")
[plt.plot(smooth(worker, 100), color="C1", alpha=0.2) for worker in test_rewards[1:3]]

# Plot Partition 2
plt.plot(smooth(test_rewards[3], 100), label="Network Partition 2", color="C2")
[plt.plot(smooth(worker, 100), color="C2", alpha=0.2) for worker in test_rewards[4:6]]

plt.title("Worker Learning Curves")
plt.legend(loc="lower right")
plt.xlabel("Episode #")
plt.ylabel("Success %")
plt.savefig("results/test_network_partition/test_network_partition.png")