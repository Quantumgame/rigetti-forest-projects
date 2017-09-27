import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
import pyquil.api as qvm
qvm_connection = qvm.SyncConnection()
#run QAOA on a square ring w/ 4 nodes at the corners

square_ring = [(0,1), (1,2), (2,3), (3,0)]
#defines graph on which to run maxcut

steps = 2
inst = maxcut_qaoa(graph=square_ring, steps=steps)
betas, gammas = inst.get_angles()
#run optimization routine on qvm

t = np.hstack((betas, gammas))
param_prog = inst.get_parametrized_program()
prog = param_prog(t)
wf, _ = qvm_connection.wavefunction(prog)
wf = wf.amplitudes
#evaluate function w/ QVM, see final state

for state_index in range(2**inst.n_qubits):
    print((inst.states[state_index], np.conj(wf[state_index],*wf[state_index])))
#wf is a numpy array of complex-valued amplitudes for each computational basis state... calculate probability
#and visualize distribution
