#!/usr/bin/python

from brian2 import *

# Parameters
C = 200 * pF
gL = 10 * nS
taum = C / gL
EL = -58 * mV
VT = -50 * mV
DeltaT = 2 * mV
Vcut = 0 * mV

# Pick an electrophysiological behaviour
tauw, a, b, Vr = 120*ms, 2*nS, .100*nA, -46*mV

eqs = """
dvm/dt = (gL*(EL - vm) + gL*DeltaT*exp((vm - VT)/DeltaT) + I - w)/C : volt
dw/dt = (a*(vm - EL) - w)/tauw : amp
I : amp
"""

neuron = NeuronGroup(1, model=eqs, threshold='vm>Vcut',
                     reset="vm=Vr; w+=b")
neuron.vm = EL
trace = StateMonitor(neuron, 'vm', record=0)
spikes = SpikeMonitor(neuron)

neuron.I = .400*nA
run(1000 * ms)

# We draw nicer spikes
vm = trace[0].vm[:]
for t in spikes.t:
    i = int(t / defaultclock.dt)
    vm[i] = 20*mV

plot(trace.t / ms, vm / mV)
xlabel('time (ms)')
ylabel('membrane potential (mV)')
show()
