import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from utils.utils import get_function_representation

class JerbiModel(nn.Module):
    '''
    Creates a parametrized quantum circuit based on the 'Parametrized quantum policies for reinforcement learning' paper by Sofiene Jerbi.
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self, 
                n_qubits,
                n_layers, 
                device,
                shots,
                diff_method, 
                entanglement,
                entanglement_pattern, 
                entanglement_gate, 
                input_scaling, 
                input_init, 
                weight_init, 
                measure):
        super(JerbiModel, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.shots = shots
        self.diff_method = diff_method
        self.entanglement = entanglement
        self.entanglement_pattern = entanglement_pattern
        self.entanglement_gate = entanglement_gate
        self.input_scaling = input_scaling
        self.input_init = input_init
        self.weight_init = weight_init
        self.measure = measure

        self.circuit = self.generate_circuit()
    
    def generate_circuit(self):
        # Call the error handling function
        self.handle_errors_and_warnings()

        # Initialize the device
        if self.shots is None:
            dev = qml.device(self.device, wires=self.n_qubits)
        else:
            dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)

        # Weight initialization
        self.weight_shapes = {
            "input_params": (self.n_layers, self.n_qubits, 2),
            "params": (self.n_layers + 1, self.n_qubits, 2)
        }
        
        self.init_method = {
            "input_params": self.input_init,
            "params": self.weight_init
        }

        @qml.qnode(dev, interface='torch', diff_method=self.diff_method)
        def qnode(inputs, params, input_params):

            # Apply Hadamard gates to all qubits
            qml.broadcast(qml.Hadamard, wires=range(self.n_qubits), pattern="single")

            # Apply layers and entanglement
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.RZ(params[layer][wire][0], wires=wire)
                    qml.RY(params[layer][wire][1], wires=wire)

                if self.entanglement:
                    qml.broadcast(self.entanglement_gate, wires=range(self.n_qubits), pattern=self.entanglement_pattern)

                # Input scaling
                if self.input_scaling is True:
                    for wire in range(self.n_qubits):
                        qml.RY(input_params[layer][wire][0] * inputs[wire], wires=wire)
                        qml.RZ(input_params[layer][wire][1] * inputs[wire], wires=wire)
                else:
                    for wire in range(self.n_qubits):
                        qml.RY(inputs[wire], wires=wire)
                        qml.RZ(inputs[wire], wires=wire)

            # Final layer
            for wire in range(self.n_qubits):
                qml.RZ(params[-1][wire][0], wires=wire)
                qml.RY(params[-1][wire][1], wires=wire)

            return self.measure(self.n_qubits)

        self.qnode = qnode
        model = qml.qnn.TorchLayer(qnode, weight_shapes=self.weight_shapes, init_method=self.init_method)

        return model

    def forward(self, inputs):
        ''' 
        Gives inputs to the circuit and outputs the respective output
        '''
        return self.circuit(inputs)
     
    def visualize_circuit(self):
        '''
        Draws the circuit
        '''
        inputs = torch.tensor([0.1 * i for i in range(self.n_qubits)], dtype=torch.float32)
        
        initialized_params = {}
        for key, shape in self.weight_shapes.items():
            initialized_params[key] = self.init_method[key](torch.empty(shape))

        # Draw the circuit
        qml.draw_mpl(self.qnode)(inputs, 
                                initialized_params["params"], 
                                initialized_params["input_params"])

    def handle_errors_and_warnings(self):
        ''' 
        Handles the errors and warnings
        '''
        # Check if the number of layers is valid
        if self.n_layers < 1:
            raise ValueError("Number of layers must be at least 1.") 

    def get_parameters(self):
        # Extract relevant attributes for JSON serialization
        return {
            "Number of Qubits": self.n_qubits,
            "Number of Layers": self.n_layers,
            "Device": str(self.device),  # Convert to string representation
            "Shots": self.shots,
            "Differentiation Method": self.diff_method,
            "Entanglement": self.entanglement,
            "Entanglement Pattern": self.entanglement_pattern,
            "Entanglement Gate": get_function_representation(self.entanglement_gate),  # Use the helper function to represent the gate
            "Input Scaling": self.input_scaling,
            "Input Initialization": get_function_representation(self.input_init),  # Use the helper function for the initializer
            "Parameters Initialization": get_function_representation(self.weight_init),  # Use the helper function for the initializer
            "Measurement Function": get_function_representation(self.measure)  # Use the helper function for the measurement function
        }
    
    @classmethod
    def info(cls):
        '''
        Provides a summary of the JerbiModel class, including its parameters and methods.
        '''
        info_text = """
        JerbiModel: A Quantum Neural Network model that creates a Parameterized Quantum Circuit based on the design in https://doi.org/10.48550/arXiv.2103.05577.

        Parameters:
        ----------
        n_qubits (int): 
            Number of qubits used in the quantum circuit.
        
        n_layers (int): 
            Number of layers in the quantum circuit. Each layer typically consists of parameterized rotations followed by entanglement gates.
        
        device (str): 
            The quantum device used for simulation or execution (e.g., 'default_qubit', 'lightning.qubit', 'lightning.gpu').

        shots (int, optional): 
            Number of times the circuit gets executed (repeated measurements). If None, the circuit is executed with analytic calculations (no shot noise).
        
        diff_method (str): 
            Differentiation method used for training the model. Common options are 'best', 'parameter-shift', 'backprop', etc.

        entanglement (bool):
            If True, entanglement between qubits is implemented. The entanglement pattern and gate are defined by `entanglement_pattern` and `entanglement_gate`, respectively.
        
        entanglement_pattern (str): 
            Entanglement pattern used in the circuit, such as 'chain', 'ring', 'all_to_all', etc., as defined by qml.broadcast patterns.
        
        entanglement_gate (function): 
            Quantum gate used for entanglement, such as qml.CZ or qml.CNOT. This gate is applied between qubits according to the specified entanglement pattern.
        
        input_scaling (bool): 
            If True, input parameters are scaled by additional learnable parameters (input_params). The input is multiplied by these parameters before being applied to the qubits.
        
        input_init (function): 
            Function to initialize the input scaling parameters, such as torch.nn.init.uniform_, torch.nn.init.ones_, or any function defined by the user.
        
        weight_init (function): 
            Function to initialize the weights of the quantum circuit, such as torch.nn.init.uniform_, torch.nn.init.normal_, or any function defined by the user.
        
        measure (function): 
            Measurement function that takes the number of qubits as an argument and returns the measurement result. Common choices are `measure_probs`, `two_measure_expval`, or any user-defined measurement function.

        Methods:
        --------
        generate_circuit(self): 
            Generates and initializes the quantum circuit based on the parameters.
        
        forward(self, inputs): 
            Takes inputs and passes them through the quantum circuit to get the output.

        visualize_circuit(self): 
            Visualizes the generated quantum circuit for the given number of qubits using the initial parameters. Useful for debugging or analyzing the circuit design.

        handle_errors_and_warnings(self): 
            Handles common errors and warnings, such as invalid parameter values, unsupported devices, and incompatible differentiation methods.
            
        """
        return info_text
    
class TfqTutorial(nn.Module):
    '''
    Creates a parameterized quantum circuit based on the TensorFlow Quantum tutorial in https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self, 
                n_qubits,
                n_layers, 
                device,
                shots, 
                diff_method, 
                entanglement,
                entanglement_pattern, 
                entanglement_gate, 
                input_scaling, 
                input_init, 
                weight_init, 
                measure):
        super(TfqTutorial, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.shots = shots
        self.diff_method = diff_method
        self.entanglement = entanglement
        self.entanglement_pattern = entanglement_pattern
        self.entanglement_gate = entanglement_gate
        self.input_scaling = input_scaling
        self.input_init = input_init
        self.weight_init = weight_init
        self.measure = measure

        self.circuit = self.generate_circuit()
    
    def generate_circuit(self):
        # Call the error handling function
        self.handle_errors_and_warnings()

        # Initialize the device
        if self.shots is None:
            dev = qml.device(self.device, wires=self.n_qubits)
        else:
            dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)
        
        # Weight initialization
        self.weight_shapes = {
            "input_params": (self.n_layers, self.n_qubits, 1),
            "params": (self.n_layers + 1, self.n_qubits, 3)
        }
        
        self.init_method = {
            "input_params": self.input_init,
            "params": self.weight_init,
        }
        
        @qml.qnode(dev, interface='torch', diff_method=self.diff_method)
        def qnode(inputs, params, input_params):
            
            # Apply layers and entanglement
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.RX(params[layer][wire][0], wires=wire)
                    qml.RY(params[layer][wire][1], wires=wire)
                    qml.RZ(params[layer][wire][2], wires=wire)

                if self.entanglement:
                    qml.broadcast(self.entanglement_gate, wires=range(self.n_qubits), pattern=self.entanglement_pattern)

                # Input scaling
                if self.input_scaling:
                    for wire in range(self.n_qubits):
                        qml.RX(input_params[layer][wire][0] * inputs[wire], wires=wire)
                else:
                    for wire in range(self.n_qubits):
                        qml.RX(inputs[wire], wires=wire)
                
            # Final layer
            for wire in range(self.n_qubits):
                qml.RX(params[-1][wire][0], wires=wire)
                qml.RY(params[-1][wire][1], wires=wire)
                qml.RZ(params[-1][wire][2], wires=wire)

            return self.measure(self.n_qubits)

        self.qnode = qnode

        model = qml.qnn.TorchLayer(qnode, weight_shapes=self.weight_shapes, init_method=self.init_method)  
        
        return model
    
    def forward(self, inputs):
        ''' 
        Gives inputs to the circuit and outputs the respective output
        '''
        return self.circuit(inputs)
    
    def visualize_circuit(self):
        '''
        Draws the circuit
        '''
        inputs = torch.tensor([0.1 * i for i in range(self.n_qubits)], dtype=torch.float32)
        
        initialized_params = {}
        for key, shape in self.weight_shapes.items():
            initialized_params[key] = self.init_method[key](torch.empty(shape))

        # Draw the circuit
        qml.draw_mpl(self.qnode)(inputs, 
                                initialized_params["params"], 
                                initialized_params["input_params"])

    def handle_errors_and_warnings(self):
        ''' 
        Handles the errors and warnings
        '''
        # Check if the number of layers is valid
        if self.n_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

    def get_parameters(self):
        # Extract relevant attributes for JSON serialization
        return {
            "Number of Qubits": self.n_qubits,
            "Number of Layers": self.n_layers,
            "Device": str(self.device),  # Convert to string representation
            "Shots": self.shots,
            "Differentiation Method": self.diff_method,
            "Entanglement": self.entanglement,
            "Entanglement Pattern": self.entanglement_pattern,
            "Entanglement Gate": get_function_representation(self.entanglement_gate),  # Use the helper function to represent the gate
            "Input Scaling": self.input_scaling,
            "Input Initialization": get_function_representation(self.input_init),  # Use the helper function for the initializer
            "Parameters Initialization": get_function_representation(self.weight_init),  # Use the helper function for the initializer
            "Measurement Function": get_function_representation(self.measure)  # Use the helper function for the measurement function
        }        
    
    @classmethod
    def info(cls):
        '''
        Provides a summary of the TFQ class, including its parameters and methods.
        '''
        info_text = """
        TFQ circuit: A Parameterized Quantum Circuit based on the design in the TensorFlow Quantum tutorial in https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning

        Parameters:
        ----------
        n_qubits (int): 
            Number of qubits used in the quantum circuit.
        
        n_layers (int): 
            Number of layers in the quantum circuit. Each layer typically consists of parameterized rotations followed by entanglement gates.
        
        device (str): 
            The quantum device used for simulation or execution (e.g., 'default_qubit', 'lightning.qubit').

        shots (int, optional): 
            Number of times the circuit gets executed (repeated measurements). If None, the circuit is executed with analytic calculations (no shot noise).
        
        diff_method (str): 
            Differentiation method used for training the model. Common options are 'best', 'parameter-shift', 'backprop', etc.

        entanglement (bool):
            If True, entanglement between qubits is implemented. The entanglement pattern and gate are defined by `entanglement_pattern` and `entanglement_gate`, respectively.
        
        entanglement_pattern (str): 
            Entanglement pattern used in the circuit, such as 'chain', 'ring', 'all_to_all', etc., as defined by qml.broadcast patterns.
        
        entanglement_gate (function): 
            Quantum gate used for entanglement, such as qml.CZ or qml.CNOT. This gate is applied between qubits according to the specified entanglement pattern.
        
        input_scaling (bool): 
            If True, input parameters are scaled by additional learnable parameters (input_params). The input is multiplied by these parameters before being applied to the qubits.
        
        input_init (function): 
            Function to initialize the input scaling parameters, such as torch.nn.init.uniform_, torch.nn.init.ones_, or any function defined by the user.
        
        weight_init (function): 
            Function to initialize the weights of the quantum circuit, such as torch.nn.init.uniform_, torch.nn.init.normal_, or any function defined by the user.
        
        measure (function): 
            Measurement function that takes the number of qubits as an argument and returns the measurement result. Common choices are `measure_probs`, `two_measure_expval`, `three_measure_expval`, or any user-defined measurement function.

        Methods:
        --------
        generate_circuit(self): 
            Generates and initializes the quantum circuit based on the parameters.
        
        forward(self, inputs): 
            Takes inputs and passes them through the quantum circuit to get the output.

        visualize_circuit(self): 
            Visualizes the generated quantum circuit for the given number of qubits using the initial parameters. Useful for debugging or analyzing the circuit design.

        handle_errors_and_warnings(self): 
            Handles common errors and warnings, such as invalid parameter values, unsupported devices, and incompatible differentiation methods.
            
        """
        return info_text
    
class UQC(nn.Module):
    '''
    Creates a parameterized quantum circuit based on the 'Data re-uploading for a universal quantum classifier' paper by Adrián Pérez-Salinas.
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self, 
                n_qubits, 
                n_layers, 
                state_dim,
                device,
                shots, 
                diff_method,
                encoding_type,
                entanglement,
                entanglement_pattern, 
                entanglement_gate, 
                input_init,
                weight_init,
                bias_init,
                measure):
        super(UQC, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.device = device
        self.shots = shots
        self.diff_method = diff_method
        self.encoding_type = encoding_type
        self.entanglement = entanglement
        self.entanglement_pattern = entanglement_pattern
        self.entanglement_gate = entanglement_gate
        self.input_init = input_init
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.measure = measure
        self.input_scaling = None

        self.circuit = self.generate_circuit()
    
    def generate_circuit(self):
        if self.shots is None:
            dev = qml.device(self.device, wires=self.n_qubits)
        else:
            dev = qml.device(self.device, wires=self.n_qubits, shots=self.shots)
        
        if self.encoding_type == 'full':
            self.weight_shapes = {
                "input_params": (self.n_layers, self.n_qubits, self.state_dim),
                "params": (self.n_layers, self.n_qubits, 1),
                "bias": (self.n_layers, self.n_qubits)
            }
        elif self.encoding_type == 'partial':
            self.weight_shapes = {
            "input_params": (self.n_layers, self.n_qubits, int(self.state_dim/self.n_qubits)),
            "params": (self.n_layers, self.n_qubits, 1),
            "bias": (self.n_layers, self.n_qubits)
            }
        
        self.init_method = {
            "input_params": self.input_init,
            "params": self.weight_init,
            "bias": self.bias_init
        }
        
        @qml.qnode(dev, interface='torch', diff_method=self.diff_method)
        def qnode(inputs, input_params, params, bias):

            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    if self.encoding_type == 'full':
                        hadamard_product = torch.dot(inputs.clone().detach(), input_params[layer][wire])
                        angle = hadamard_product + bias[layer][wire]
                    elif self.encoding_type == 'partial':
                        separate_inputs = np.array_split(inputs,self.n_qubits)
                        hadamard_product = torch.dot(separate_inputs[wire], input_params[layer][wire])
                        angle = hadamard_product + bias[layer][wire]

                    qml.RZ(angle, wires=wire)
                    
                    qml.RY(params[layer][wire][0], wires=wire)
                    
                if self.entanglement:
                    qml.broadcast(self.entanglement_gate, wires=range(self.n_qubits), pattern=self.entanglement_pattern)

            return self.measure(self.n_qubits)

        self.qnode = qnode

        model = qml.qnn.TorchLayer(self.qnode, weight_shapes=self.weight_shapes, init_method=self.init_method)
        
        return model

    def forward(self, inputs):
        ''' 
        Gives inputs to the circuit and outputs the respective output
        '''
        return self.circuit(inputs)
    
    def visualize_circuit(self):
        inputs = torch.tensor([0.1 * i for i in range(self.state_dim)], dtype=torch.float32)
        
        # Initialize all parameters using the provided initialization methods
        initialized_params = {}
        for key, shape in self.weight_shapes.items():
            initialized_params[key] = self.init_method[key](torch.empty(shape))

        # Draw the circuit
        qml.draw_mpl(self.qnode)(inputs, 
                                initialized_params["weights"], 
                                initialized_params["params"], 
                                initialized_params["bias"])

    def handle_errors_and_warnings(self):
        ''' 
        Handles the errors and warnings
        '''
        # Check if the number of layers is valid
        if self.n_layers < 1:
            raise ValueError("Number of layers must be at least 1.")
        
    def get_parameters(self):
        # Extract relevant attributes for JSON serialization
        return {
            "Number of Qubits": self.n_qubits,
            "Number of Layers": self.n_layers,
            "State Dimension": self.state_dim,
            "Device": str(self.device),  # Convert to string representation
            "Shots": self.shots,
            "Differentiation Method": self.diff_method,
            "Encoding Type": self.encoding_type,
            "Entanglement": self.entanglement,
            "Entanglement Pattern": self.entanglement_pattern,
            "Entanglement Gate": get_function_representation(self.entanglement_gate),  # Use the helper function to represent the gate
            "Input Initialization": get_function_representation(self.input_init),  # Use the helper function for the initializer
            "Parameters Initialization": get_function_representation(self.weight_init),  # Use the helper function for the initializer
            "Bias Initialization": get_function_representation(self.bias_init),  # Use the helper function for the initializer
            "Measurement Function": get_function_representation(self.measure)  # Use the helper function for the measurement function
        }
    
    @classmethod
    def info(cls):
        '''
        Provides a summary of the UQC class and its parameters/methods.
        '''
        info_text = """
        Creates a parameterized quantum circuit based on the work in 'Data re-uploading for a universal quantum classifier' by Adrián Pérez-Salinas.

        Parameters:
        ----------
        n_qubits (int): 
            Number of qubits used in the quantum circuit.

        n_layers (int): 
            Number of layers in the quantum circuit. Each layer consists of parameterized rotations and entanglement gates.

        state_dim (int): 
            Dimensionality of the state space, determining the size of the weights associated with each qubit.

        device (str): 
            The quantum device to be used for execution, such as 'default.qubit', 'lightning.qubit', etc.

        shots (int, optional): 
            Number of times the circuit gets executed (repeated measurements). If None, the circuit is executed with analytic calculations (no shot noise).

        diff_method (str): 
            Differentiation method used for training the model. Common options include 'best', 'parameter-shift', 'adjoint', and 'backprop'.

        encoding_type (str): 
            Type of encoding used for the input data. Can be 'full' for complete encoding or 'partial' for partial encoding, which changes the shape of weights.

        entanglement (bool): 
            If True, entanglement between qubits is implemented. The entanglement pattern and gate are defined in entanglement_pattern and entanglement_gate, respectively.

        entanglement_pattern (str): 
            Entanglement pattern used in the circuit, such as 'chain', 'ring', 'all_to_all', etc., as defined by qml.broadcast patterns.

        entanglement_gate (function): 
            Quantum gate used for entanglement, such as qml.CZ or qml.CNOT. This gate is applied between qubits according to the specified entanglement pattern.

        input_init (function): 
            Function to initialize the weights of the quantum circuit, such as torch.nn.init.uniform_, torch.nn.init.normal_, or a user-defined function.

        weight_init (function): 
            Function to initialize the parameters of the quantum circuit, similar to input_init.

        bias_init (function): 
            Function to initialize the bias terms in the quantum circuit, such as torch.nn.init.uniform_, torch.nn.init.zeros_, or a user-defined function.

        measure (function): 
            Measurement function that takes the number of qubits as an argument and returns the measurement result. Common choices are measure_probs, two_measure_expval, or any user-defined measurement function.
        
        Methods:
        --------
        generate_circuit(self): 
            Generates and initializes the quantum circuit based on the parameters.
        
        forward(self, inputs): 
            Takes inputs and passes them through the quantum circuit to get the output.

        visualize_circuit(self): 
            Visualizes the generated quantum circuit for the given number of qubits using the initial parameters. Useful for debugging or analyzing the circuit design.

        handle_errors_and_warnings(self): 
            Handles common errors and warnings, such as invalid parameter values, unsupported devices, and incompatible differentiation methods.
        """
        return info_text