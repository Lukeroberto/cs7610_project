import matplotlib.pyplot as plt
import numpy as np

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
