import numpy as np
from matplotlib import pyplot as plt

# Define the LIF neuron model
def LIF(v, t, input_current, membrane_capacitance, membrane_resistance):
    dvdt = (input_current - (v/membrane_resistance))/membrane_capacitance
    return dvdt

# We import scipy ordinary differential equations solver
from scipy.integrate import odeint

#Defined the model parameters
membrane_capacitance = 1.
membrane_resistance = 1.
input_current = 1.

# Initial value of the neuron membrane potential
v0 = 10.

# Simulation timesteps
T  = 30.0                             # final time (the units will depend on what units we have chosen for the model constants)
N = 10000                             # number of time-steps for the simulation
t = np.linspace(0.,T, N)

# Call differential equation solver and stores result in variable res
solution_membrane_potential = odeint(LIF, v0, t, args = (input_current, membrane_capacitance, membrane_resistance))

# Plot results
plt.figure(figsize=(10, 6))
plt.grid()
plt.title("LIF neuron", fontsize=16)
plt.plot(t, solution_membrane_potential)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Neuron membrane potential', fontsize=16)
plt.show()