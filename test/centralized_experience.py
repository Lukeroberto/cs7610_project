import ray 
import psutil
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import sys
import torch
from tqdm import tqdm as tqdm


from src.utils.environments import *
from src.utils.dqn import CentralizedRunner, DQNAgent_central
import src.utils.plotting_utils  as plotting_utils


def main():
    # Number of physical cores on machine
    num_cpus = psutil.cpu_count(logical=False)

    # Start ray
    ray.init(logging_level="ERROR")

    N_EPISODES = int(sys.argv[1])
    num_agents = num_cpus if (sys.argv[2] == "max") else int(sys.argv[2])

    print(f"Num runners: {num_agents}")
    print(f"Num episodes: {N_EPISODES}")

    # Initialize workers
    env = ContinuousGridWorld()
    AGENT = DQNAgent_central(env)
    TARGET_UPDATE_INTERVAL = 30
    BATCH_SIZE = 50
    steps_per_cycle = 50
    
    runners = [CentralizedRunner.remote(ContinuousGridWorld(), i) for i in range(num_agents)]

    obj_ids = [runner.get_experience.remote(dict(AGENT.model.named_parameters()), BATCH_SIZE) for runner in runners]
    REWARDS = []
    for i in tqdm(range(1,N_EPISODES)):
        for _ in range(num_agents):
            ready, not_ready = ray.wait(obj_ids, timeout=1)
            if len(ready) == 0: continue
            tmp_reward = 0
            for r in ready:
                batch, reward, actor_id = ray.get(r)
                AGENT.add_batch(batch)
                eps = max(0.5*(1-i/N_EPISODES),0)
                not_ready.append(runners[actor_id].get_experience.remote(dict(AGENT.model.named_parameters()), BATCH_SIZE, eps))
                tmp_reward += reward
                obj_ids = not_ready
                continue
        REWARDS.append(tmp_reward/len(ready))
        for _ in range(steps_per_cycle):
            AGENT.optimize()
        if i % TARGET_UPDATE_INTERVAL == 0:
            AGENT.update_target()

    success_rate = sum([AGENT.run_episode() for _ in range(100)])/100
    print(f"Success rate: {success_rate}")

    plt.figure()
    # plt.plot(REWARDS)
    plt.plot(np.convolve(REWARDS, np.ones(100)/100.,mode='valid'))
    plt.show()



    # plotting_utils.plot_workers(rewards, smoothing=200)
    # plt.savefig("results/aggregate/workers.png")
    # plotting_utils.plot_workers_aggregate(rewards, smoothing=200)
    # plt.savefig("results/aggregate/workers_agg.png")

    # plt.show()

    # print("Done!")

if __name__ == "__main__":
    main()