from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks.sampler_qnn import SamplerQNN
from qiskit.quantum_info import Statevector
from sspace import QCkt
from qiskit.algorithms.optimizers import COBYLA
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import numpy as np
import time




class QAE:
    """
    Class that is responsible for running a generated
    circuit and returning the results of fidelity on
    the test data provided
    """
    def __init__(self, train_set, test_set, nlatent, ntrash):
        self.train_set = train_set
        self.test_set = test_set
        self.nlatent = nlatent
        self.ntrash = ntrash
        self.run_hist = []
        self.metrics = {}

    def _qae_ckt(self, sampled_qnn):
        qubits = self.nlatent + 2*self.ntrash + 1
        qr = QuantumRegister(qubits, "q")
        cr = ClassicalRegister(1, "c")

    
        circuit = QuantumCircuit(qr, cr)
        circuit = circuit.compose(sampled_qnn, 
                                  range(self.nlatent + self.ntrash))
        circuit.barrier()
        aux_qubit = self.nlatent + 2*self.ntrash
        circuit.h(aux_qubit)
        for i in range(self.ntrash):
            # control qubit, wire1, wire2
            circuit.cswap(aux_qubit, 
                          self.nlatent + i, self.nlatent + self.ntrash + i)
        circuit.h(aux_qubit)
        circuit.measure(aux_qubit, cr[0])
        return circuit

    def _make_train_ckt(self, sampled_qnn):
        '''circuit without the inverse to train weights of AE'''
        qc = QuantumCircuit(self.nlatent + 2*self.ntrash + 1, 1)
        ae = self._qae_train_ckt(sampled_qnn=sampled_qnn)
        fm = RawFeatureVector(2**(self.nlatent + self.ntrash))
        qc = qc.compose(fm, range(self.nlatent + self.ntrash))
        qc = qc.compose(ae)
        return qc, fm, ae
    
    def _make_test_ckt(self, sampled_qnn, fm):
        '''Circuit with inverse to obtain output state'''
        test_qc = QuantumCircuit(self.nlatent + self.ntrash)
        test_qc = test_qc.compose(fm)
        test_qc = test_qc.compose(sampled_qnn)
        test_qc.barrier()
        test_qc.reset(self.ntrash+1)
        test_qc.reset(self.ntrash+2)
        test_qc = test_qc.compose(sampled_qnn.inverse())
        return test_qc


    def _optimize(self, qnn, maxiters=300, init_pt=None):
        _opt = COBYLA(maxiter=maxiters)
        def cost_fn(params):
            probs = qnn(self.train_set[0], params)
            cost = np.sum(probs[:, 1]) / len(self.train_set[0])
            self.run_hist.append(cost)
            return cost
        
        start = time.time()
        opt_res = _opt.minimize(fun=cost_fn, x0=init_pt)
        elapsed = time.time() - start
        print(f"Fit in {elapsed:0.2f} s")
        self.metrics['train_time'] = elapsed
        return opt_res
    
    def _get_fidelity(self, inp_state, out_state):
        fid = lambda x, y: np.sqrt(np.dot(x.conj(), y)**2)
        fid_vec = [fid(i, o) for i, o in zip(inp_state, out_state)]
        return np.mean(fid_vec)

    def get_metrics(self, metric_str, sampled_ckt, max_iters=300, init_pt=None):

        def identity_interpret(x):
            return x
        
        train_ckt, fm, ae = self._make_train_ckt(sampled_qnn=sampled_ckt)
        qnn = SamplerQNN(circuit=train_ckt,
                         input_params=fm.parameters,
                         weight_params=ae.parameters,
                         interpret=identity_interpret,
                         output_shape=2
                         )
        
        opt_res = self._optimize(qnn, max_iters, init_pt=init_pt)

        test_ckt = self._make_test_ckt(sampled_ckt, fm)
        tparams = [np.concatenate(td, opt_res.x) 
                   for td in self.test_set[0]]
        odata = [Statevector(test_ckt.assign_parameters(tp)).data
                    for tp in tparams]
        
        idata = [Statevector(td).data for td in self.test_set[0]]

        
        self.metrics['fidelity'] = self._get_fidelity(idata, odata)

        assert metric_str in self.metrics, f"No such metric {metric_str}"
        return self.metrics[metric_str]

    

    
    