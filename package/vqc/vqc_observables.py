import pennylane as qml

def measure_probs(qubits):
    return qml.probs(wires=range(qubits)) 

def two_measure_expval(qubits):
    pauli_string = qml.PauliZ(0)
    for i in range(1, qubits):
        pauli_string = pauli_string @ qml.PauliZ(i)
    
    expvals = []
    expvals.append(qml.expval(pauli_string))
    expvals.append(qml.expval(-pauli_string))

    return expvals

def three_measure_expval(qubits):
    expvals = []

    if qubits == 1:
        first_observable = qml.PauliZ(0)
        middle_observable = qml.PauliX(0)
        last_observable = -qml.PauliZ(0)
    elif qubits == 2:
        first_observable = qml.PauliZ(0)
        middle_observable = qml.PauliZ(0) @ qml.PauliZ(1) 
        last_observable = qml.PauliZ(1)       
    elif qubits >= 4:
        first_observable = qml.PauliZ(0)
        middle_observable = qml.PauliZ(1)
        for i in range(2, qubits - 1):
            middle_observable = middle_observable @ qml.PauliZ(i)
        last_observable = qml.PauliZ(qubits - 1)                    
    else:
        raise ValueError("Unsupported number of qubits: only 1, 3, or 4 qubits are supported")

    expvals.append(qml.expval(first_observable))
    expvals.append(qml.expval(middle_observable))
    expvals.append(qml.expval(last_observable))

    return expvals
