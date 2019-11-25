import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import *
from src.utils.dqn import DQNAgent_solo
from tqdm import tqdm

def main(n_episodes=2000, n_repeats=5, save=True, show=False):
    env = ContinuousGridWorld()
    agent = DQNAgent_solo(env, [0,1,2])

    N_EPISODES = n_episodes
    N_REPEATS = n_repeats
    agent.set_scheduler((0, N_EPISODES), (0.5, 0.01))

    RETURNS = np.zeros((N_REPEATS,N_EPISODES))
    for repeat_id in range(N_REPEATS):
        agent.reset_model()
        for ep_id in tqdm(range(N_EPISODES)):
            RETURNS[repeat_id, ep_id] = agent.run_episode(ep_id)
    if save:
        np.save("results/baseline/single_agent.npy", RETURNS)
    
    if show:
        plt.figure()
        for i in range(N_REPEATS):
            plt.plot(np.convolve(RETURNS[i,:], 0.02*np.ones(50), mode='valid'))

        plt.show()

if __name__ == "__main__":
    main()