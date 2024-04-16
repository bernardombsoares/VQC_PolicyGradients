import pennylane as qml
import torch
from pqc_operations import *
#from measures import measure_probs, measure_expval_pairs


def jerbi_circuit(n_qubits, n_layers, shots, input_scaling, diff_method, weight_init, input_init, measure, measure_qubits):

    if shots is None:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    
    if n_layers < 1:
        raise ValueError("Number of layers can't take values below 1")
    
    weight_shapes = {"params": (n_layers + 1, n_qubits, 2),
                    "input_params": (n_layers, n_qubits, 2)}
    init_method   = {"params": weight_init,
                    "input_params": input_init}

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def qnode(inputs, params, input_params):
        
    #in case n_qubits != input length
        if n_qubits > len(inputs) and n_qubits % len(inputs) == 0:
            multiplier = n_qubits // len(inputs)
            inputs = torch.cat([inputs] * multiplier)
        elif n_qubits != len(inputs) and n_qubits % len(inputs) != 0:
            raise ValueError('Number of qubits cannot be divided by input lenght')


        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        
        for layer in range(n_layers):
            for wire in range(n_qubits):
                RZRY(wire, params[layer][wire])

            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")

            if input_scaling:
                for wire in range(n_qubits):
                    RYRZ_encode(wire, inputs[wire], input_params[layer][wire])
            else:
                for wire in range(n_qubits):
                    RYRZ_encode(wire, inputs[wire])

        for wire in range(n_qubits):
            RZRY(wire, params[-1][wire])

        return measure(measure_qubits)

    model = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)  
    
    return model