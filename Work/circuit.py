import pennylane as qml
import numpy as np
import torch
from torch.nn.parameter import Parameter

def create_zz_operator(n_qubits):
    ZZ = qml.PauliZ(0)
    for i in range(1, n_qubits):
        ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))
    return ZZ

class CircuitGenerator:

    def __init__(self, n_qubits, n_layers, shots):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots


    def jerbi(self, init_params, params, measure_type = 'probs', observables = None):

        dev = qml.device('default.qubit', wires=self.n_qubits, shots=None, analytic=False)
        weight_shapes = {
                            "init_params": (self.n_layers, 2, self.n_qubits),
                            "params": (self.n_layers + 1, 2, self.n_qubits)
                        }

        init_method = {
                            "init_params": torch.nn.init.normal_,
                            "params": torch.nn.init.uniform,
                        }

        '''
        measure_type should be either 'probs' (probability of each bitstring) or 'expval' (expectation value )
        params should have shape=(n_layers + 1, 2, n_qubits)
        init_params should have shape=(n_layers, 2, n_qubits)
        '''

        if measure_type != 'probs' or measure_type != 'expval':
            return 'Error: provided measure_type is not valid.'
        
        qml.broadcast(qml.Hadamard, wires=range(self.n_qubits), pattern="single")
        for i in range(self.n_layers):
            qml.broadcast(qml.RZ, wires=range(self.n_qubits), pattern="single", parameters=params[i][0])
            qml.broadcast(qml.RY, wires=range(self.n_qubits), pattern="single", parameters=params[i][1])

            qml.broadcast(qml.CNOT, wires=range(self.n_qubits), pattern="chain")

            qml.broadcast(qml.RY, wires=range(self.n_qubits), pattern="single", parameters=init_params[i][0])
            qml.broadcast(qml.RZ, wires=range(self.n_qubits), pattern="single", parameters=init_params[i][1])

        qml.broadcast(qml.RZ, wires=range(self.n_qubits), pattern="single", parameters=params[self.n_layers][0])
        qml.broadcast(qml.RY, wires=range(self.n_qubits), pattern="single", parameters=params[self.n_layers][1])
        
        if measure_type == 'probs':
            if observables == None:
                qcircuit = qml.probs(wires=self.n_qubits)
            else:
                qcircuit = qml.probs(op=observables)

        elif measure_type == 'expval':
            if observables == None:
                observables = create_zz_operator(self.n_qubits)
                qcircuit = qml.expval(op=observables)
            else:
                qcircuit = qml.expval(op=observables)

        circuit = qml.QNode(qcircuit, dev, interface="torch", diff_method="backprop")
        model = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes, init_method=init_method)

        return model



    