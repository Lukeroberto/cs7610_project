import ray 
import psutil
import numpy as np
import numpy.random as npr

from src.utils.environments import RandomizedCartpole

# Number of physical cores on machine
num_cpus = psutil.cpu_count(logical=False)

# Start ray
ray.init()

@ray.remote(num_cpus=num_cpus)
def explore(id, env, steps):
    env.reset()
    num_steps = 0
    for i in range(steps):
        state, _, done, _ = env.step(npr.randint(env.nA))
        num_steps += 1
        # print(f"Proc-{id}: State -> ({state})")

        if done:
            break
    print(f"Proc-{id}: Total steps = {num_steps}")
    

# Initialize workers
ids = list()
MAX_STEPS = 30
for i in range(num_cpus):
    id = explore.remote(i, RandomizedCartpole(), MAX_STEPS)
    ids.append(id)

# Wait for procs to finish
ray.wait(ids)