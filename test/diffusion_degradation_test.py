
import psutil

from src.utils.decentralized_dqn import *
from src.utils.environments import *
from src.utils.graph_utils import *
from src.utils.plotting_utils import *
from src.utils.parser import *


# Number of physical cores on machine
NUM_CORES = psutil.cpu_count(logical=False)

# Start ray
ray.init(logging_level="ERROR")

p = test_parser()
args = p.parse_args()

NUM_EPISODES = int(args.length)

# One of these architectures
spoke = spoke_adj(NUM_CORES)
chain = chain_adj(NUM_CORES)

graph = chain

# Initialize workers
agents = [DQNAgent.remote(ContinuousGridWorld(), [i, args.test], opt=False) for i in range(NUM_CORES)]

# Set neighbors
neighbors = generate_neighbor_graph(graph, agents)
for agent in agents:
    agent.set_neighbors.remote(neighbors[agent])

# Set center model
for agent in agents:
    agent.load_torch_file.remote("results/example_agent.pth")

# Train 
rewards = np.zeros((len(agents), NUM_EPISODES))
for ep_id in range(NUM_EPISODES):
    temp = []
    for agent_id, agent in enumerate(agents):
        agent.diffuse.remote(ep_id)
        temp.append(agent.eval_episode.remote())
    
    temp = [ray.get(t) for t in temp]
    rewards[:, ep_id] = temp


plot_workers(rewards)
plt.savefig(f"results/test_{args.test}/workers.png")

np.save(f"results/test_{args.test}/worker_rewards.npy", rewards)
plt.show()
# np.save(f"results/test_{args.test}/worker_diffusions.npy", diffusions)