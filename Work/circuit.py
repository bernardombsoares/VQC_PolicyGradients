import pennylane as qml
import numpy as np


def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)


def jerbi_layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])




class CircuitGenerator:
    def __init__(self, n_qubits, n_layers=None, shots=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device("default.qubit.torch", wires=n_qubits, shots=shots)


    def jerbi(self):
        @qml.qnode(self.dev)
        def qnode(self,init_params,params):
            for layer_idx in range(self.n_layers):
                if (layer_idx == 0):
                    encode(self.n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure(n_qubits)

    model = qml.qnn.TorchLayer(circuit, shapes)

    return model


    def simplified_two_design(self, init_params, params, measurement_qubits=0, prod_approx=False):

        qml.SimplifiedTwoDesign(initial_layer_weights=init_params, weights=params, wires=range(self.n_qubits))
    




    def special_unitary(self, params):
        return 'a'
    





    def create_circuit(self, circ="simplified_two_design", fim=False):
        if circ == "rzry":
            qcircuit = self.RZRY
        elif circ == "simplified_two_design":
            qcircuit = self.simplified_two_design
        elif circ == "special_unitary":
            qcircuit = self.special_unitary
        # Define other circuit architectures similarly...

        if not fim:
            circuit = qml.QNode(qcircuit, self.dev, interface="torch", diff_method="backprop")
        else:
            circuit = qml.QNode(qcircuit, self.dev)

        return circuit