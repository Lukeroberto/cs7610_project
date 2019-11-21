import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import *
from src.utils.dqn import DQNAgent_solo
from tqdm import tqdm

env = ContinuousGridWorld()
agent = DQNAgent_solo(env, 1)

N_EPISODES = 2000
agent.set_scheduler((0, N_EPISODES), (0.5, 0.01))

RETURNS = np.zeros(N_EPISODES)
for ep_id in tqdm(range(N_EPISODES)):
    RETURNS[ep_id] = agent.run_episode(ep_id)

# N = 20
# ind = np.dstack(np.meshgrid(np.arange(N), np.arange(N))).reshape(-1,2)
# q_vals = agent.model.forward(env.torch_state(ind/N + 0.5/N)).detach().numpy()
# vals = q_vals.max(axis=1)
# actions = q_vals.argmax(axis=1)

# grid = vals.reshape(N,N)
# plt.figure()
# plt.imshow(np.log(grid), cmap="Greys", origin="lower")
# for (x,y), a in zip(ind,actions):
#     plt.arrow(y-0.25*N*env._steps[a][1],
#                 x-0.25*N*env._steps[a][0],
#                 0.4*N*env._steps[a][1], 
#                 0.4*N*env._steps[a][0], 
#                 color="r", head_width=0.2)
# plt.plot(N*env.goal[1], N*env.goal[0], 'go')
# plt.axis('off')
    

plt.figure()
# plt.plot(RETURNS)
plt.plot(np.convolve(RETURNS, 0.02*np.ones(50), mode='valid'))
plt.ioff()
plt.show()