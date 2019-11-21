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

def full_adj(length):
    adj = np.ones((length, length), dtype=np.int)
    diag = np.diag_indices(length)
    adj[diag] = 0

    return adj

def spoke_adj(length):
    adj = np.zeros((length, length), dtype=np.int)

    adj[0, 1:] = 1
    adj[1:, 0] = 1

    return adj

def network_partition():
    # 2 triangles connected
    adj = np.zeros((6,6), dtype=np.int)

    adj[0, [1, 2]] = 1
    adj[1, [0, 2]] = 1
    adj[2, [0, 1, 3]] = 1
    adj[3, [2, 4, 5]] = 1
    adj[4, [3, 5]] = 1
    adj[5, [3, 4]] = 1

    return adj