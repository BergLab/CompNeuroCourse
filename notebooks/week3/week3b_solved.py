#%% 
from brian2 import *
import brian2 as b2
from matplotlib import pyplot
import numpy as np

seed = 123
np.random.seed(seed)
b2.seed(seed)
print(f"\nSeed for simulation: {seed}\n") 

b2.defaultclock.dt = 1 * b2.ms


#%% 
# start_scope()

# N = 100

# lif_dynamics = """
# dv/dt = -(v-v_rest) / membrane_time_scale : volt (unless refractory) 
# """
# firing_threshold = 20 * b2.mV
# abs_refractory_period = 2 * b2.ms
# v_reset = 0 * b2.mV # reset voltage after firingv_rest = 0 * b2.mV  # rest voltage to each neuron, ie. constant input voltage
# v_rest = 25 * b2.mV 
# membrane_time_scale = 20 * b2.ms
# synaptic_delay = 1.5 * b2.ms 

# G  = NeuronGroup(N=N, model=lif_dynamics, threshold="v>firing_threshold",
#                 reset="v=v_reset", refractory=abs_refractory_period,
#                 method="exact")

# G.v = np.random.uniform(low=0, high=20, size=N)*mV

# S = Synapses(source=G, target=G, model= 'w : volt', on_pre="v += w", delay=synaptic_delay)
# S.connect(p=1)
# S.w = np.random.normal(0, 1, size=(N,N)).flatten()*mV
# # S.w = np.random.uniform(low=-10, high=10, size=(N,N)).flatten()*mV

# SpikeMon = SpikeMonitor(G)

# run(200*ms)

# figure(figsize=(16,8))
# subplot(121)
# plot(SpikeMon.t/ms, SpikeMon.i, '.k')
# xlabel('Time (ms)')
# ylabel('Neuron index')
# pyplot.show()

#%% 
start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

N = 100

G = NeuronGroup(N, eqs, threshold='v>1', reset='v = 0', method='exact')
G.I = np.random.rand(N)
G.tau = 10*ms
G.v = np.random.uniform(low=0, high=2, size=N)


S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(p=1.0)
S.w = np.random.uniform(low=-1, high=1, size=(N,N)).flatten()

StateMon = StateMonitor(G, 'v', record=[0,1,2])
SpikeMon = SpikeMonitor(G)

run(100*ms)

figure(figsize=(16,8))
subplot(121)
plot(SpikeMon.t/ms, SpikeMon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')

subplot(122)
plot(StateMon.t/ms, StateMon.v[0], label='Neuron 0')
plot(StateMon.t/ms, StateMon.v[1], label='Neuron 1')
plot(StateMon.t/ms, StateMon.v[2], label='Neuron 2')
xlabel('Time (ms)')
ylabel('v')
legend(fontsize=16)
pyplot.show()
