import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def plot_workers(arrays, smoothing=1):
    plt.figure()

    [plt.plot(smooth(worker, smoothing)) for worker in arrays]
    plt.title("Worker Learning Curves")
    plt.legend([f"worker_{i}" for i in range(len(arrays))])

def plot_workers_aggregate(arrays, smoothing=50):
    arrays = np.array(arrays)
    worker_mean = arrays.mean(axis=0)
    worker_std = arrays.std(axis=0)

    worker_mean = smooth(worker_mean, smoothing)
    worker_std = smooth(worker_std, smoothing)

    eps = np.array(range(len(worker_mean))) 

    fig, ax = plt.subplots(1)
    ax.plot(eps, worker_mean)
    ax.fill_between(eps, worker_mean+worker_std, worker_mean-worker_std, alpha=0.5)
    plt.title("Aggregate Learning Curve")

def plot_training_progress(rewards, diffusions, ids):
    p_id = ids[0]
    test_id = ids[1]
    ep_id = ids[2]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate [%]', color='b')
    ax1.plot(smooth(rewards[:ep_id], 100), color='b')
    ax1.tick_params(axis='y', colors='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Diffusions', color='r')
    ax2.plot(smooth(diffusions[:ep_id], 100), color='r')
    ax2.tick_params(axis='y', colors='r')
    fig.tight_layout() 

    results_dir = f"results/test_{test_id}/" 
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + f"/agent_{p_id}_returns.png")
    plt.close()

