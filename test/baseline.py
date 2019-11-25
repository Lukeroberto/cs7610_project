import numpy as np 
import matplotlib.pyplot as plt 

from test import single_agent, centralized_experience

single_agent.main()
centralized_experience.main()

plt.figure()
centralized = np.load("centralized_experience_6runners.npy")
single = np.load("single_agent.npy")

data = [centralized, single]
labels = ["centralized", "single"]

W = 100
for d in data:
    result = np.zeros((len(d), len(d[0])-W+1))
    for i in range(len(d)):
        result[i] = np.convolve(d[i], np.ones(W)/W, mode="valid")
    avg = np.mean(result, axis=0)
    std = np.std(result, axis=0) 
    plt.plot(avg)
    plt.fill_between(np.arange(result.shape[1]), avg-std, avg+std, alpha=0.3)

plt.legend(labels)
plt.show()