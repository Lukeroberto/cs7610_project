import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import *
from src.utils.dqn import DQNAgent_solo
from tqdm import tqdm

env = ContinuousGridWorld()
agent = DQNAgent_solo(env, 1)

N_EPISODES = 2000
agent.set_scheduler((0, N_EPISODES-50), (1, 0.01))

RETURNS = np.zeros(N_EPISODES)
for ep_id in tqdm(range(N_EPISODES)):
    RETURNS[ep_id] = agent.run_episode(ep_id)
    

plt.figure()
plt.plot(np.convolve(RETURNS, 0.01*np.ones(100), mode='valid'))
plt.ioff()
plt.show()