
import ray 
import psutil
import numpy as np
import numpy.random as npr

from src.utils.environments import RandomizedCartpole
from src.utils.dqn import DQNAgent

# Number of physical cores on machine
num_cpus = psutil.cpu_count(logical=False)

# Start ray
ray.init(logging_level="ERROR")

# Config
EPSILON = 0.5
BATCH_SIZE = 16
TRAINING_BATCHES = 100

@ray.remote(num_cpus=num_cpus)
def train(id, env):

    # Setup env and agent
    env.reset()
    agent = DQNAgent(id, env, EPSILON)
    for i in range(TRAINING_BATCHES):
        agent.optimize(BATCH_SIZE)
    
    print(f"Proc-{id}: Training Complete")

    

# Initialize workers
ids = list()
print(f"Num cpus: {num_cpus}")
for i in range(num_cpus):
    id = train.remote(i, RandomizedCartpole())
    ids.append(id)

# Wait for procs to finish
ray.wait(ids)
