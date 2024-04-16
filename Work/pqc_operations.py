import pennylane as qml

def RZRY(qubit, params):
    qml.RZ(param[0], wires=qubit)
    qml.RY(param[1], wires=qubit)

def RZRY_encode(qubit, inputs, params = None):
    if params is None:
        qml.RZ(inputs, wires=qubit)
        qml.RY(inputs, wires=qubit)
    else:
        qml.RZ(params[0]*inputs, wires=qubit)
        qml.RY(params[1]*inputs, wires=qubit)

def RYRZ(qubit, params):
    qml.RY(param[0], wires=qubit)
    qml.RZ(param[1], wires=qubit)

def RYRZ_encode(qubit, inputs, params = None):
    if params is None:
        qml.RY(inputs, wires=qubit)
        qml.RZ(inputs, wires=qubit)
    else:
        qml.RY(params[0]*inputs, wires=qubit)
        qml.RZ(params[1]*inputs, wires=qubit)