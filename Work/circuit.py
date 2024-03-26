import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils import measure_selection

def jerbi_model(n_qubits, n_layers, weight_init, measure_type, observables):

    dev = qml.device("default.qubit", wires=n_qubits)

    observables = observables if observables is not None else None
    
    weight_shapes = {"params": (n_layers + 1, n_qubits, 2)}
    init_method   = {"params": weight_init}

    @qml.qnode(dev, interface='torch')
    def qnode(inputs, params):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RZ(params[layer][wire][0], wires=wire)
                qml.RY(params[layer][wire][1], wires=wire)

            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")
            for wire in range(n_qubits):
                qml.RY(inputs[wire], wires=wire)
                qml.RZ(inputs[wire], wires=wire)

        for wire in range(n_qubits):
            qml.RZ(params[-1][wire][0], wires=wire)
            qml.RY(params[-1][wire][1], wires=wire)
    
        return measure_selection(n_qubits, measure_type, observables)

    model = qml.qnn.TorchLayer(qnode, weight_shapes=weight_shapes, init_method=init_method)

    return model



####################################################################################################

class CircuitGenerator(nn.Module):

    def __init__(self, 
                 n_qubits, 
                 n_layers,  
                 shots, 
                 weight_init=torch.nn.init.uniform_, 
                 input_init = torch.nn.init.ones_, 
                 measure_type = 'probs', 
                 observables = None):
        super(CircuitGenerator, self).__init__()
        self.n_qubits = n_qubits                        #number of qubits
        self.n_layers = n_layers                        #number of layers
        self.shots = shots                              #number of shots
        self.measure_type = measure_type                #measure type - 'probs' or 'expval'
        self.observables = observables                  #observables if the used wants
        self.weight_init = weight_init                  #weight initialization method
        self.input_init = input_init                    #input weight initialization method



    def jerbi_model(self, input_scaling = False):
        
        if input_scaling:
            self.input_params = Parameter(torch.empty(self.n_layers, self.n_qubits, 2))
            self.input_init(self.input_params)
        else:
            self.input_params = None
            
        self.q_layers = jerbi_model(n_qubits=self.n_qubits,
                                    n_layers=self.n_layers,
                                    weight_init = self.weight_init,
                                    measure_type=self.measure_type,
                                    observables=self.observables)
        
        return self.q_layers

    def jerbi_input(self,inputs):

        if self.input_params is not None:
            inputs = inputs * self.input_params

        outputs = self.q_layers(inputs)

        return outputs


    