import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import odeint

# Define the firing rate model
def firing_rate_model(y, t, tau, w, J):
    # y represents the firing rates of the neurons
    # Define the differential equation
    dy_dt = (-y + np.dot(w, np.tanh(y)) + J) / tau
    return dy_dt

# Parameters
N = 100  # Number of neurons
tau = 10.0  # Time constant
w = np.random.randn(N, N) / np.sqrt(N)  # Recurrent weight matrix
J = np.random.randn(N)  # External input current

# Initial conditions
y0 = np.random.rand(N)

# Time points
t = np.linspace(0, 100, 1000)

# Solve the differential equation
solution = odeint(firing_rate_model, y0, t, args=(tau, w, J))

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(min(5, N)):  # Plot the first 5 neurons to avoid clutter
    plt.plot(t, solution[:, i], label=f'Neuron {i+1}')
plt.xlabel('Time')
plt.ylabel('Firing Rate')
plt.title('Firing Rate Model of a Recurrent Network')
plt.legend()
plt.show()



# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges based on the weight matrix
for i in range(N):
    for j in range(N):
        if w[i, j] != 0:  # If there is a connection
            G.add_edge(i, j, weight=w[i, j])

# Draw the network
plt.figure(figsize=(12, 6))
pos = nx.circular_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000,
        edge_color='gray', width=[abs(w[i][j]) for i, j in G.edges()])
plt.title('Network Structure')
plt.show()
