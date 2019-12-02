
@ray.remote
class Agent():
    def run(self):
        return 1

# create remote agent
agent = Agent.remote()

# run method
p_id = agent.run.remote()

# wait till run is complete (BLOCKING)
ready = ray.wait(p_id, timeout=1)

# access the output
result = ray.get(ready)



