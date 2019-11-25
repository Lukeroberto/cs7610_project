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


def main(n_runners=6, n_repeats=5, n_episodes=2000, save=True, show=False):
    # Number of physical cores on machine
    num_cpus = psutil.cpu_count(logical=False)

    # Start ray
    ray.init(logging_level="ERROR")

    N_EPISODES = n_episodes
    N_REPEATS = 5

    # Initialize workers
    env = ContinuousGridWorld()
    AGENT = DQNAgent_central(env)
    TARGET_UPDATE_INTERVAL = 30
    EP_LENGTH = 50
    steps_per_cycle = 50
    
    runners = [CentralizedRunner.remote(ContinuousGridWorld(), i) for i in range(num_runners)]

    obj_ids = [runner.get_experience.remote(dict(AGENT.model.named_parameters()), EP_LENGTH) for runner in runners]
    REWARDS = [[] for a in range(N_REPEATS)]
    for repeat_id in range(N_REPEATS):
        AGENT.reset_model()
        for i in tqdm(range(1,N_EPISODES+1)):
            for _ in range(n_runners):
                ready, not_ready = ray.wait(obj_ids, timeout=1)
                if len(ready) == 0: continue
                tmp_reward = 0
                avg_n_steps = 0
                for r in ready:
                    batch, reward, actor_id = ray.get(r)
                    AGENT.add_batch(batch)
                    eps = max(0.5*(1-i/N_EPISODES),0)
                    not_ready.append(runners[actor_id].get_experience.remote(dict(AGENT.model.named_parameters()), EP_LENGTH, eps))
                    tmp_reward += reward
                    avg_n_steps += len(batch["R"])
                    obj_ids = not_ready
                    continue
            REWARDS[repeat_id].append(tmp_reward/len(ready))
            # avg_n_steps /= len(ready)
            for _ in range(int(avg_n_steps)):
                AGENT.optimize()
            if i % TARGET_UPDATE_INTERVAL == 0:
                AGENT.update_target()

    if show:
        plt.figure()
        for r in range(N_REPEATS):
            plt.plot(np.convolve(REWARDS[r], np.ones(100)/100.,mode='valid'))
        plt.show()
    if save:
        max_length = max([len(a) for a in REWARDS])
        reward_array = np.zeros((N_REPEATS, max_length))
        for r in range(N_REPEATS):
            reward_array[r] = REWARDS[r][:max_length]
        np.save(f"results/baseline/centralized_experience_{n_runners}runners.npy", reward_array)


if __name__ == "__main__":
    main()