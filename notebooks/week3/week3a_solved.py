##%
figure(figsize=(18,6))
subplot(121)
plot(spikemon.t/ms, spikemon.i, '.k')
xlabel('Time [ms]')
ylabel('Neuron index')
subplot(122)
plot(statemon.t/ms, statemon.v[0])
plot(statemon.t/ms, statemon.v[1])
plot(statemon.t/ms, statemon.v[2])
xlabel('Time (ms)')
ylabel('Membrane potential');