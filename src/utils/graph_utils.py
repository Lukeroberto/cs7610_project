import numpy as np

def generate_neighbor_graph(adj_mat, agents):
    """
    Generates a dictionary of agents to their neighbors

    """
    neighbor_dict = {}
    for i, row in enumerate(adj_mat):
        neighbor_dict[agents[i]] = []
        for j, el in enumerate(row):
            if el == 1:
                neighbor_dict[agents[i]].append(agents[j])

    return neighbor_dict

########
# Different adjacency matricies

def chain_adj(length):
    adj = np.zeros((length, length), dtype=np.int)

    # Top and bottom off-diagonal
    for i in range(length - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1

    return adj

