import ray
import psutil 

num_cpus = psutil.cpu_count()
ray.init()

@ray.remote(num_cpus=num_cpus)
def alive(id):
    print(f"Process-{id} Alive!")
    return 1

for i in range(num_cpus):
    id = alive.remote(i)
    ray.get(id)
    