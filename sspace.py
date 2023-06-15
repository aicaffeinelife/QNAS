import os
from typing import *
import networkx as nx
import numpy as np
from qiskit_gate_maps import *
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal



class QCkt:
    """A simplified version of QGraph class
    that generates circuits based on random
    rotation and entangling gates at runtime 
    for deployment on device. 
    """
    def __init__(self,  n_qubits):
        self.qubits = n_qubits
    
    
    def sample_random_circuit(self, num_rots, 
                              num_ents, n_reps, ent_style='full'):
        
        rot_gates = np.random.choice(ROT_OPS, size=num_rots).tolist()
        ent_gates = np.random.choice(ENT_OPS, size=num_ents).tolist()
        
        circuit = TwoLocal(self.qubits, 
                           rotation_blocks=rot_gates,
                           entanglement_blocks=ent_gates,
                           entanglement=ent_style,
                           insert_barriers=True, reps=n_reps)
        return circuit
    
    def gen_ckt_with_specs(self, rot_gates, ent_gates, ent_map, n_reps):
        """Accept a modification and generate a twolocal circuit """
        pass