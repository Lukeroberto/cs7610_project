import numpy as np 
import matplotlib.pyplot as plt 
import datetime
import ray
from tqdm import tqdm as tqdm
import pickle

from src.utils.environments import ContinuousGridWorld
from src.utils.decentralized_dqn import DQNAgent
from src.utils.centralized_dqn import DQNAgent_central, CentralizedRunner
from src.utils.graph_utils import *
from src.utils.dqn_utils import train

def main(N_EPISODES=2000, load=False):
    ray.init(logging_level="ERROR")
    
    if load:
        with open('results/baseline/data.pickle', 'rb') as handle:
            DATA = pickle.load(handle)
    else:
        DATA = dict()
        
        ######################
        ## train single agent
        ######################
        DATA["single"] = dict()
        env = ContinuousGridWorld()
        agent = DQNAgent.remote(env, [0,1,2], logging=False)
        agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.01))
        DATA["single"]["start"] = datetime.datetime.now()
        DATA["single"]["returns"] = np.zeros((5,N_EPISODES))
        for trial_id in range(5):
            agent.reset_model.remote()
            for ep_id in tqdm(range(N_EPISODES)):
                tmp = agent.run_episode.remote(ep_id)
                DATA["single"]["returns"][trial_id, ep_id] = ray.get(tmp)
        DATA["single"]["end"] = datetime.datetime.now()
        
        ##########################
        ## train centralized agent
        ##########################

        N_RUNNERS = 12
        TARGET_UPDATE_INTERVAL = 30
        EP_LENGTH = 50

        DATA["central"] = dict()
        env = ContinuousGridWorld()
        driver = DQNAgent_central(env)
        runners = [CentralizedRunner.remote(ContinuousGridWorld(), i) for i in range(N_RUNNERS)]
        obj_ids = [runner.get_experience.remote(dict(driver.model.named_parameters()), EP_LENGTH) for runner in runners]

        DATA["central"]["start"] = datetime.datetime.now()
        DATA["central"]["returns"] = np.zeros((5, N_EPISODES))
        for trial_id in range(5):
            driver.reset_model()
            for ep_id in tqdm(range(1,N_EPISODES+1)):
                for _ in range(N_RUNNERS):
                    ready, not_ready = ray.wait(obj_ids, timeout=1)
                    if len(ready) == 0: continue
                    tmp_reward = 0
                    avg_n_steps = 0
                    for r in ready:
                        batch, reward, actor_id = ray.get(r)
                        driver.add_batch(batch)
                        eps = max(0.5*(1-ep_id/N_EPISODES),0)
                        not_ready.append(runners[actor_id].get_experience.remote(dict(driver.model.named_parameters()), EP_LENGTH, eps))
                        tmp_reward += reward
                        avg_n_steps += len(batch["R"])
                        obj_ids = not_ready
                        continue
                DATA["central"]["returns"][trial_id, ep_id-1] = tmp_reward/len(ready)
                for _ in range(int(avg_n_steps)):
                    driver.optimize()
                if ep_id % TARGET_UPDATE_INTERVAL == 0:
                    driver.update_target()
        DATA["central"]["end"] = datetime.datetime.now()


        ###########################
        ###### train decentralized 
        ############################
        
        N_AGENTS = 12
        fully_connected = full_adj(N_AGENTS)

        # Initialize workers
        agents = [DQNAgent.remote(ContinuousGridWorld(), [i, ''], logging=False) for i in range(N_AGENTS)]
        [agent.set_scheduler.remote((0, N_EPISODES), (0.5, 0.01)) for agent in agents]

        # Set neighbors
        neighbors = generate_neighbor_graph(fully_connected, agents)
        for agent in agents:
            agent.set_neighbors.remote(neighbors[agent])

        # Train 
        DATA["decentral"] = dict()
        DATA["decentral"]["start"] = datetime.datetime.now()
        DATA["decentral"]["returns"] = np.zeros((5, N_EPISODES))
        for trial_id in range(5):
            [agent.reset_model.remote() for agent in agents]
            train(agents, N_EPISODES)
            DATA["decentral"]["returns"][trial_id] = np.mean(ray.get([agent.get_returns.remote() for agent in agents]),axis=0)
        DATA["decentral"]["end"] = datetime.datetime.now()

        
        with open("results/baseline/data.pickle", "wb") as handle:
            pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # end if

    W = 100
    plt.figure()
    for i, k in enumerate(DATA.keys()):
            z = DATA[k]["returns"]
            z_conv = np.zeros((5, N_EPISODES-W+1))
            for j in range(5):
                z_conv[j] = np.convolve(z[j], np.ones(W)/W, mode='valid')
            avg = np.mean(z_conv, axis=0)
            std = np.std(z_conv, axis=0)
            plt.plot(avg)
            plt.fill_between(np.arange(len(avg)), avg+std, avg-std, alpha=0.3)
    
    plt.legend(DATA.keys())
    plt.savefig("results/baseline/example.png")


if __name__ == "__main__":
    main()