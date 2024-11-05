import torch
import torch.nn as nn

class QuantumPolicy(nn.Module):
    '''
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self, circuit, post_processing):
        super(QuantumPolicy, self).__init__()
        self.circuit = circuit
        self.post_processing = post_processing

    def sample(self, inputs):
        '''
        Samples an action from the action probability distribution
        '''
        policy = self.forward(inputs)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action), policy
    
    def forward(self, inputs):
        '''
        Input state is fed to the circuit - its output is then fed to the post processing 
        '''
        probs = self.circuit.forward(inputs)
        probs_processed = self.post_processing.forward(probs)
        return probs_processed

    @classmethod
    def info(cls):
        '''
        Provides a summary of the QuantumPolicy class and its parameters/methods.
        '''
        info_text = """

        Combines a quantum circuit for generating action probabilities with a post-processing step to create a valid probability distribution for action selection.

        Parameters:
        ----------
        circuit (object): 
            A quantum circuit instance that processes input states and generates raw probabilities.
        
        post_processing (object): 
            A post-processing instance that transforms raw probabilities into a valid probability distribution.

        Methods:
        -------
        sample(inputs):
            Samples an action based on the computed probability distribution for a given input.
        
        forward(inputs):
            Computes the forward pass of the policy network by processing the inputs through the circuit 
            and the post-processing module.
        """
        return info_text