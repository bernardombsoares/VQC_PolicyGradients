import torch
import torch.nn as nn

class QuantumPolicyModel(nn.Module):
    
    def __init__(self, circuit, policy):
        super(QuantumPolicyModel, self).__init__()
        self.circuit = circuit
        self.policy = policy

    def forward(self, inputs):
        '''
        Input state is fed to the circuit - its output is then fed to the post processing 
        '''
        probs = self.circuit.input(inputs)
        probs_processed = self.policy.input(probs)
        return probs_processed
    
    def sample(self, inputs):
        '''
        Samples an action from the action probability distribution aka policy
        '''
        policy = self.forward(inputs)
        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action), policy
    
    def T_schedule(self,current_step,total_steps):

        max_T = 1.0  # Initial temperature
        min_T = 0.1  # Final temperature
        decay_rate = 0.0025  # Decay rate

        self.policy.T = max_T * (1 - decay_rate * current_step / total_steps)
        self.policy.T = max(self.policy.T, min_T)

    def get_parameters(self):
        '''
        Returns the values of each set of parameters
        '''
        parameter_values = [param.clone().detach().numpy().flatten() for param in self.circuit.parameters()]
        return parameter_values
    
    def get_gradients(self):
        '''
        Returns the gradient values of each set of parameters
        '''
        gradients = [torch.flatten(param.grad.clone().detach()) if param.grad is not None else torch.flatten(torch.zeros_like(param)) for name, param in self.circuit.named_parameters()]
        return gradients