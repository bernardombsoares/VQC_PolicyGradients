import pennylane as qml
import torch
from pqc_operations import *
#from measures import measure_probs, measure_expval_pairs

class PQC():

    def __init__(self, n_qubits, n_layers, shots, input_scaling, diff_method, weight_init, input_init, measure):

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.input_scaling = input_scaling
        self.diff_method = diff_method
        self.weight_init = weight_init
        self.input_init = input_init
        self.measure = measure


    def jerbi_circuit(self):

        if self.shots is None:
            dev = qml.device("default.qubit", wires=self.n_qubits)
        else:
            dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        
        if self.n_layers < 1:
            raise ValueError("Number of layers can't take values below 1")
        
        weight_shapes = {"params": (self.n_layers + 1, self.n_qubits, 2),
                        "input_params": (self.n_layers, self.n_qubits, 2)}
        init_method   = {"params": self.weight_init,
                        "input_params": self.input_init}

        @qml.qnode(dev, interface='torch', diff_method=self.diff_method)
        def qnode(inputs, params, input_params):
            
            #in case n_qubits != input length
            if self.n_qubits > len(inputs) and self.n_qubits % len(inputs) == 0:
                multiplier = self.n_qubits // len(inputs)
                inputs = torch.cat([inputs] * multiplier)
            elif self.n_qubits != len(inputs) and self.n_qubits % len(inputs) != 0:
                raise ValueError('Number of qubits cannot be divided by input length')

            qml.broadcast(qml.Hadamard, wires=range(self.n_qubits), pattern="single")
            
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    RZRY(wire, params[layer][wire])

                qml.broadcast(qml.CNOT, wires=range(self.n_qubits), pattern="chain")

                if self.input_scaling:
                    for wire in range(self.n_qubits):
                        RYRZ_encode(wire, inputs[wire], input_params[layer][wire])
                else:
                    for wire in range(self.n_qubits):
                        RYRZ_encode(wire, inputs[wire])

            for wire in range(self.n_qubits):
                RZRY(wire, params[-1][wire])

            return self.measure

        model = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)  
        
        return model
