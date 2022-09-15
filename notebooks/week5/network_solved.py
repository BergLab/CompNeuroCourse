#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%%
sps_directed = []
sps_undirected = []
sizes = []

for size in range(10, 200):
    adj_matrix = np.random.randint(low=0, high=2, size=(size, size)) # create a square random matrix with 0s and 1s of of size networkSize
    G_directed = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
    sps_directed.append(nx.average_shortest_path_length(G_directed))
    G_undirected = nx.from_numpy_matrix(adj_matrix)
    sps_undirected.append(nx.average_shortest_path_length(G_undirected))
    sizes.append(size)

plt.figure(figsize=(8,12))
plt.plot(sizes, sps_undirected, label='Undirected G')
plt.plot(sizes, sps_directed, label='Directed G')
plt.legend(fontsize=12)


#%%

sps_undirected = []
size = 100
ps = []

for p in np.arange(0.1, 2, 0.05):

    G_undirected = nx.gnp_random_graph(size, p, directed=False)
    sps_undirected.append(nx.average_shortest_path_length(G_undirected))

    ps.append(p)

plt.figure(figsize=(8,12))
plt.plot(ps, sps_undirected, label='Undirected G')
plt.legend(fontsize=12)
