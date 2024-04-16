import pennylane as qml

def measure_probs(qubits):
    ''' 
    Outputs the probability of each bitstring of size 'qubits'.
    '''
    return qml.probs(wires=range(qubits)) 


def measure_expval_pairs(qubits):
    ''' 
    Outputs the expectation value of the tensor product of Pauli Z operators acting on pairs of qubits.
    '''
    expvals = []
    for i in range(qubits // 2):
        expvals.append(qml.expval(qml.PauliZ(2*i) @ qml.PauliZ(2*i + 1)))
    return expvals

