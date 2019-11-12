import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import RandomizedCartpole
from src.utils.dqn import DQNAgent_solo
from tqdm import tqdm

env = RandomizedCartpole()
agent = DQNAgent_solo(env, 1)

N_EPISODES = 500
agent.set_scheduler((0, N_EPISODES-50), (0.2, 0.00))

RETURNS = np.zeros(N_EPISODES)
for ep_id in tqdm(range(N_EPISODES)):
    RETURNS[ep_id] = agent.run_episode(ep_id)
    

plt.figure()
plt.plot(RETURNS)

plt.ioff()
plt.show()