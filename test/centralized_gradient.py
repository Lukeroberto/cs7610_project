import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sys
import torch
from tqdm import tqdm as tqdm

from src.utils.environments import *
from src.utils.dqn import CentralizedRunner, MLP_DQN
import src.utils.plotting_utils  as plotting_utils

def apply_gradient(model, optimizer, gradients):
    for g, p in zip(gradients, model.parameters()):
        if g is not None:
            p.grad = torch.from_numpy(g)
    optimizer.step()

def main():
    # Number of physical cores on machine
    num_cpus = psutil.cpu_count(logical=False)

    # Start ray
    ray.init(logging_level="ERROR")


    # i have gotten it to work for 20k ep, 10 workers
    N_EPISODES = int(sys.argv[1])
    num_agents = num_cpus if (sys.argv[2] == "max") else int(sys.argv[2])

    print(f"Num runners: {num_agents}")
    print(f"Num episodes: {N_EPISODES}")

    # Initialize workers
    gamma = 0.98
    lr = 1e-4

    agents = [CentralizedRunner.remote(ContinuousGridWorld(),i, gamma, lr) for i in range(num_agents)]

    env = ContinuousGridWorld()
    model = MLP_DQN(env.state_dim, env.nA)
    target = MLP_DQN(env.state_dim, env.nA)
    target.load_state_dict(model.state_dict())
    TARGET_UPDATE_INTERVAL = 10

    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    batch_size = 64

    obj_ids = [agent.calc_gradient.remote(dict(model.named_parameters()),
                                            dict(target.named_parameters()),
                                             batch_size) for agent in agents]
    REWARDS = []
    for i in tqdm(range(N_EPISODES)):
        ready, not_ready = ray.wait(obj_ids, timeout=1)
        if len(ready) == 0: continue
        tmp = 0
        for r in ready:
            grads, reward, actor_id = ray.get(r)
            tmp+=reward
            apply_gradient(model, optimizer, grads)
            eps = 0.5*(1-i/N_EPISODES)
            not_ready.append(agents[actor_id].calc_gradient.remote(dict(model.named_parameters()),
                                                            dict(target.named_parameters()), 
                                                            batch_size, eps))
            obj_ids = not_ready
            continue
        REWARDS.append(tmp/len(ready))
        if i % TARGET_UPDATE_INTERVAL == 0:
            target.load_state_dict(model.state_dict())
        
    plt.figure()
    plt.plot(np.convolve(REWARDS, np.ones(200)/200.,mode='valid'))
    plt.show()


    # plotting_utils.plot_workers(rewards, smoothing=200)
    # plt.savefig("results/aggregate/workers.png")
    # plotting_utils.plot_workers_aggregate(rewards, smoothing=200)
    # plt.savefig("results/aggregate/workers_agg.png")

    # plt.show()

    # print("Done!")

if __name__ == "__main__":
    main()