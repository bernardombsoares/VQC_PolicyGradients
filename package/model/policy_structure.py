import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyType(nn.Module):
    
    def __init__(self, n_qubits, n_actions, post_processing = 'raw_contiguous'):
        self.n_actions = n_actions
        self.n_qubits = n_qubits
        self.post_processing = post_processing
        self.T = 1

    def input(self,probs):
        if self.post_processing == 'raw_contiguous':
            policy = self.raw_contiguous(probs)
        elif self.post_processing == 'raw_parity':
            policy = self.raw_parity(probs)
        elif self.post_processing == 'softmax':
            policy = self.softmax(probs)
        else:
            raise ValueError("Invalid post-processing method specified.")
        return policy

    def raw_contiguous(self,probs):
        
        probs_flatten = probs.flatten()
        chunk_size = len(probs_flatten) // self.n_actions
        remainder = len(probs_flatten) % self.n_actions

        policy = []

        for i in range(self.n_actions):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            if i < remainder:
                end += 1

            # Update the original policy list instead of creating a new one
            policy.append(sum(probs_flatten[start:end]))

        policy_tensor = torch.stack(policy)
        return policy_tensor
        
    def raw_parity(self,probs):

        if self.n_actions % 2 != 0:
            raise ValueError('For parity-like policy, n_actions must be an even number')
        
        probs_flatten = probs.flatten()
        policy = torch.zeros(self.n_actions)
        counter = 0
        for prob in probs_flatten:
            policy[counter] += prob
            counter += 1
            if counter == self.n_actions:
                counter = 0
        
        return policy
    
    def softmax(self, probs):
        if len(probs) == self.n_actions:
            scaled_output = probs * self.T
            softmax_output = F.softmax(scaled_output, dim=0)
            return softmax_output
        else:
            probs_flatten = probs.flatten()
            chunk_size = len(probs_flatten) // self.n_actions
            remainder = len(probs_flatten) % self.n_actions

            policy = []

            for i in range(self.n_actions):
                start = i * chunk_size
                end = (i + 1) * chunk_size

                if i < remainder:
                    end += 1

                # Update the original policy list instead of creating a new one
                policy.append(sum(probs_flatten[start:end]))
            policy_tensor = torch.stack(policy)
            softmax_output = F.softmax(policy_tensor/self.T, dim=0)
            return softmax_output