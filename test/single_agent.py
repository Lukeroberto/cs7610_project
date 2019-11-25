import numpy as np 
import matplotlib.pyplot as plt

from src.utils.environments import *
from src.utils.dqn import DQNAgent_solo
from src.utils.decentralized_dqn import DQNAgent
from tqdm import tqdm

<<<<<<< Updated upstream
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
=======
env = ContinuousGridWorld()
N_EPISODES = 1200
# agent = DQNAgent_solo(env, [0, 0, 0])
agent = DQNAgent(env, [0, 0, 0], logging=False)
agent.load_torch_file("results/example_agent.pth")

RETURNS = np.zeros(N_EPISODES)
for ep_id in tqdm(range(N_EPISODES)):
    RETURNS[ep_id] = agent.eval_episode()
# agent.set_scheduler((0, N_EPISODES), (0.5, 0.01))

# RETURNS = np.zeros(N_EPISODES)
# for ep_id in tqdm(range(N_EPISODES)):
#     RETURNS[ep_id] = agent.run_episode(ep_id)

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
>>>>>>> Stashed changes
    
    if show:
        plt.figure()
        for i in range(N_REPEATS):
            plt.plot(np.convolve(RETURNS[i,:], 0.02*np.ones(50), mode='valid'))

        plt.show()

if __name__ == "__main__":
    main()