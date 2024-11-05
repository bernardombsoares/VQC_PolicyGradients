import numpy as np
import torch
import torch.nn as nn
from utils.utils import get_function_representation


class PolicyPostProcessing(nn.Module):
    '''
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self,
                 n_qubits,
                 n_actions,
                 policy_type, 
                 beta_scheduling, 
                 beta,
                 increase_rate, 
                 output_scaling,
                 output_init):
        super(PolicyPostProcessing, self).__init__()

        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.policy_type = policy_type
        self.beta_scheduling = beta_scheduling
        self.beta = beta
        self.increase_rate = increase_rate
        self.output_scaling = output_scaling
        self.output_init = output_init

        if self.output_scaling == True:
            self.output_params = nn.parameter.Parameter(torch.Tensor(self.n_actions), requires_grad=True)
            self.output_init(self.output_params)
        else:
            self.register_parameter('w_input', None)

    def forward(self,probs):
        ''' 
        Takes the VQC output and applies the selected policy 
        '''
        if self.policy_type == 'raw_contiguous':
            policy = self.raw_contiguous(probs)
        elif self.policy_type == 'raw_parity':
            policy = self.raw_parity(probs)
        elif self.policy_type == 'softmax':
            policy = self.softmax(probs)
        else:
            raise ValueError("Invalid post-processing method specified.")
        return policy

    def raw_contiguous(self,probs):
        ''' 
        Applies the Contiguous partition to the probabilities of the basis states (design to work with qml.probs)
        '''
        log_n_actions = int(np.log2(self.n_actions))
        
        # Ensure the number of actions does not exceed the number of basis states (determined by n_qubits)
        if log_n_actions > self.n_qubits:
            raise ValueError('Number of actions exceeds the number of basis states!')

        # Split the probabilities in a contiguous manner
        probs_split = torch.chunk(probs, self.n_actions)
        policy = [torch.sum(prob) for prob in probs_split]
        return(torch.stack(policy))

    def raw_parity(self,probs):
        ''' 
        Applies the Parity partition to the probabilities of the basis states (design to work with qml.probs)
        '''
        log_n_actions = int(np.log2(self.n_actions))

        # Check if the number of actions is a power of 2
        if log_n_actions < 1.0 or not (np.floor(log_n_actions) == np.ceil(log_n_actions)):
            raise NotImplementedError('Number of actions needs to be a power of two!')

        # Ensure the number of actions does not exceed the number of qubits
        if log_n_actions > self.n_qubits:
            raise ValueError('Number of actions exceeds number of basis states!')

        # Flatten the probability distribution to handle it as a single-dimensional array
        if log_n_actions == 1:
            summed_tensors = []
            even_tensor = probs[::2]  # Elements at even indices
            odd_tensor = probs[1::2]  # Elements at odd indices
            summed_tensors.append(torch.sum(even_tensor))
            summed_tensors.append(torch.sum(odd_tensor))
        else:
            probs_split = list(torch.chunk(probs, self.n_actions//2))
            summed_tensors = []

            for tensor in probs_split:
                even_tensor = tensor[::2]  # Even indexed elements
                odd_tensor = tensor[1::2]  # Odd indexed elements
                summed_tensors.append(torch.sum(even_tensor))
                summed_tensors.append(torch.sum(odd_tensor))

        return torch.stack(summed_tensors)
    
    def softmax(self, probs):
        ''' 
        Applies a softmax to the expected values of some observable
        '''
        if self.output_scaling == True:
            probs *= self.output_params

        scaled_output = probs * self.beta
        softmax_output = F.softmax(scaled_output, dim=0)
        return softmax_output
    
    def beta_schedule(self):
        '''
        Increases the inverse temperature parameter by 'increase_rate'
        '''
        if self.beta_scheduling == True and self.policy_type == 'softmax':
            self.beta += self.increase_rate

    def get_parameters(self):
        # Extract relevant attributes for JSON serialization
        return {
            "Policy Type": self.policy_type,
            "Beta Scheduling": self.beta_scheduling,
            "Beta": self.beta,
            "Increase Rate": self.increase_rate,
            "Output Scaling": self.output_scaling,
            "Output Initialization": get_function_representation(self.output_init),
        }
    
    @classmethod
    def info(cls):
        '''
        Provides a summary of the PolicyType class and its parameters/methods.
        '''
        info_text = """

        Processes the output of the circuit into one of the implemented policies (Born Contiguous, Born Parity, Softmax)

        Parameters:
        ----------
        n_actions (int): 
            Number of actions available for the agent to choose from.
        
        policy_type (str): 
            Type of policy applied to the probability distribution:
            - 'raw_contiguous': Applies the Born Contiguous-like policy.
            - 'raw_parity': Applies the Born Parity-like policy.
            - 'softmax': Applies the softmax policy to the expectation values.

        beta_scheduling (bool): 
            If True, updates the inverse temperature parameter (beta) after each episode. Used only for the softmax policy.
        
        beta (float): 
            Inverse temperature parameter used for scaling probabilities in the softmax policy.
        
        increase_rate (float): 
            Amount added to beta at the end of each episode, if beta_scheduling is True.
        
        output_scaling (bool): 
            If True, scales the output probabilities by learnable parameters.
        
        output_init (function): 
            Initialization function for output parameters, such as torch.nn.init.uniform_, torch.nn.init.ones_, etc.
        
        Methods:
        -------
        forward(self, probs):
            Selects an action based on the chosen post_processing method.
        
        raw_contiguous(self, probs):
            Sums up contiguous chunks of probabilities and returns the probability of each action.
        
        raw_parity(self, probs):
            Sums up probabilities based on parity and returns the probability of each action.
        
        softmax(self, probs):
            Applies a softmax function to the scaled probabilities and returns the probability of each action.
        
        beta_schedule(self):
            Updates the beta parameter if beta_scheduling is True. Only applicable for the softmax method.
        """
        return info_text