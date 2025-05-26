import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Lorenz system
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Parameters
sigma = 10
rho = 28
beta = 8 / 3
dt = 0.01
T = 40
t = np.arange(0, T, dt)

# Initial condition
initial_state = [1.0, 1.0, 1.0]

# Integrate the system
solution = odeint(lorenz, initial_state, t, args=(sigma, rho, beta))
x, y, z = solution.T  # Transpose to get individual arrays

# Set up 3D plot
fig = plt.figure(facecolor='black')  # Set figure background color
ax = fig.add_subplot(111, projection='3d', facecolor='black')  # Set axes background color
line, = ax.plot([], [], [], lw=0.8, color='cyan')  # Set line color

# Set axis limits
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))
ax.set_title("Lorenz Attractor (Animated with odeint)", color='white')  # Set title color

# Set grid and tick colors
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Animation update function
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,

# Create animation (no blitting for 3D!)
ani = FuncAnimation(fig, update, frames=len(t), interval=1, blit=False)

plt.show()
