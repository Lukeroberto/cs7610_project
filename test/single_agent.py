import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import *
from src.utils.dqn import DQNAgent_solo
from src.utils.decentralized_dqn import DQNAgent
from tqdm import tqdm

import ray

env = ContinuousGridWorld()
N_EPISODES = 1200
ray.init(logging_level="ERROR")
agent = DQNAgent.remote(env, [0, 0, 0], logging=False)

agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.01))

RETURNS = np.zeros(N_EPISODES)
for ep_id in tqdm(range(N_EPISODES)):
    RETURNS[ep_id] = ray.get(agent.run_episode.remote(ep_id))

# agent.save_weights()

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
# for i in range(N_REPEATS):
plt.plot(np.convolve(RETURNS, 0.02*np.ones(50), mode='valid'))

plt.show()

if __name__ == "__main__":
    main()