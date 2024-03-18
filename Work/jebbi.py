import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)


def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])


def measure(n_actions):
    return [qml.probs(wires= [i for i in range(int(np.log2(n_actions)))])]


def get_model(n_qubits, n_layers, data_reupload):
    dev = qml.device("default.qubit", wires=n_qubits)
    shapes = {
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if (layer_idx == 0) or data_reupload:
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure(n_qubits)

    model = qml.qnn.TorchLayer(circuit, shapes)

    return model


class QuantumNet(nn.Module):

    def __init__(self, n_layers, w_input, w_output, data_reupload):
        super(QuantumNet, self).__init__()
        self.n_qubits = 4
        self.n_actions = 2
        self.data_reupload = data_reupload
        self.q_layers = get_model(n_qubits=self.n_qubits,
                                  n_layers=n_layers,
                                  data_reupload=data_reupload)
        if w_input:
            self.w_input = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input, mean=0.)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output, mean=90.)
        else:
            self.register_parameter('w_output', None)

    def forward(self, inputs):
        if self.w_input is not None:
            inputs = inputs * self.w_input
        inputs = torch.atan(inputs)
        outputs = self.q_layers(inputs)
        outputs = (1 + outputs) / 2
        if self.w_output is not None:
            outputs = outputs * self.w_output
        else:
            outputs = 90 * outputs
        return outputs
     