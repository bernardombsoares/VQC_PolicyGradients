import pennylane as qml

def create_zz_operator(n_qubits):
    ZZ = qml.PauliZ(0)
    for i in range(1, n_qubits):
        ZZ = qml.operation.Tensor(ZZ, qml.PauliZ(i))
    return ZZ

def measure_selection(n_qubits, measure_type, observables):
    if measure_type == 'probs':
        if observables is None:
            return qml.probs(wires=range(n_qubits))
        else:
            return qml.probs(op=observables, wires=range(n_qubits))
    elif measure_type == 'expval':
        op = observables if observables is not None else create_zz_operator(n_qubits)
        return qml.expval(op=op)