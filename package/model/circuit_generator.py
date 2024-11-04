import torch
import torch.nn as nn
from package.pqc.vqc_designs import *
from pqc.measures import measure_probs
from pqc.pqc_operations import *

class CircuitGenerator(nn.Module):

    def __init__(self, n_qubits, n_layers,  shots= None, input_scaling=True, design='jerbi_circuit', diff_method = 'backprop', weight_init=torch.nn.init.uniform_, input_init = torch.nn.init.ones_, measure = None, measure_qubits = None):
        super(CircuitGenerator, self).__init__()
        '''

        Creates a parameterized quantum circuit based on the arguments:

            n_qubits(int) = Number of qubits
            n_layers(int) = Number of layers (0 if no data re-uploading)
            shots(int) = Number of times the circuit gets executed
            input_scaling(bool) = Input parameters are used if True (input*input_params)
            design(str) = The PQC ansatz design ('jerbi_circuit')
            diff_method(str) = Differentiation method ('best', 'backprop', 'parameter-shift', ...)
            weight_init (torch.nn.init) = How PQC weights are initialized (.uniform_, .ones_, ...)
            input_init (torch.nn.init) = How input weights are initialized (.uniform_, .ones_, ...)
            measure (function) = Measure function (measure_probs, measure_expval_pairs)
            measure_qubits (int) = Number of qubits to be measured (in some cases might be equal to the number of qubits)
            
        '''
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.input_scaling = input_scaling
        self.design = design
        self.diff_method = diff_method
        self.weight_init = weight_init
        self.input_init = input_init
        if measure is None:
            self.measure = measure_probs
        else:
            self.measure = measure

        if measure_qubits is None:
            self.measure_qubits = n_qubits
        else:
            self.measure_qubits = measure_qubits

        if self.design == 'jerbi_circuit':
            self.circuit = PQC(n_qubits = self.n_qubits,
                                        n_layers = self.n_layers,
                                        shots = self.shots,
                                        input_scaling = self.input_scaling,
                                        diff_method = self.diff_method,
                                        weight_init = self.weight_init,
                                        input_init = self.input_init,
                                        measure = self.measure,
                                        measure_qubits = self.measure_qubits).jerbi_circuit()
        else:
            raise ValueError("Unsupported circuit type")

    def input(self,inputs):

        outputs = self.circuit(inputs)
        return outputs