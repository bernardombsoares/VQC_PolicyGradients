import pennylane as qml

def measure_probs(qubits):
    '''
    Returns a list with the probability of each computational basis state
    '''
    return qml.probs(wires=range(qubits)) 

def two_measure_expval(qubits):
    '''
    Computes and returns the expectation values of two observables for a given number of qubits.

    For the specified number of qubits, this function constructs a PauliZ observable that acts
    on all qubits. It returns two expectation values:
    - The first expectation value is for the constructed PauliZ observable.
    - The second expectation value is for the negative of the constructed PauliZ observable.
    '''
    
    pauli_string = qml.PauliZ(0)
    for i in range(1, qubits):
        pauli_string = pauli_string @ qml.PauliZ(i)
    
    expvals = []
    expvals.append(qml.expval(pauli_string))
    expvals.append(qml.expval(-pauli_string))

    return expvals

def three_measure_expval(qubits):
    '''
    Computes and returns the expectation values of three observables based on the number of qubits.

    This function defines and evaluates three observables depending on the input number of qubits:
    - For 1 qubit: PauliZ, PauliX, and negative PauliZ.
    - For 2 qubits: PauliZ on the first qubit, tensor product of PauliZ on both qubits, and PauliZ on the second qubit.
    - For 3 or more qubits: PauliZ on the first qubit, a chain of PauliZ on the intermediate qubits, and PauliZ on the last qubit.
    '''

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

'''
    Users can define their own observable functions to compute different expectation values 
    for other observables.
    Write your observable functions below:
'''
