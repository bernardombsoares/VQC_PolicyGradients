import pennylane as qml

class Measure():

    def __init__(self,qubits):

        self.qubits = qubits

    def measure_probs(self):
        ''' 
        Outputs the probability of each bitstring of size 'qubits'.
        '''
        return qml.probs(wires=range(self.qubits)) 

    def measure_expval_pairs(self):
        ''' 
        Outputs the expectation value of the tensor product of Pauli Z operators acting on pairs of qubits.
        '''
        expvals = []
        for i in range(self.qubits // 2):
            expvals.append(qml.expval(qml.PauliZ(2*i) @ qml.PauliZ(2*i + 1)))
        return expvals

