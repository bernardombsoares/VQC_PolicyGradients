import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyType(nn.Module):
    
    def __init__(self, n_actions, n_qubits):
        self.n_actions = n_actions
        self.n_qubits = n_qubits

    def raw_contiguous(self,probs):

        probs_flatten = probs.flatten()
        chunk_size = len(probs_flatten) // np.log2(self.n_actions)
        remainder = len(probs_flatten) % np.log2(self.n_actions)

        policy = [0] * int(np.log2(self.n_actions))

        for i in range(int(np.log2(self.n_actions))):

            start = i * chunk_size
            end = (i + 1) * chunk_size

            if i < remainder:
                end += 1

            policy[i] = sum(probs_flatten[start:end])
            
        policy = torch.tensor(policy)

        return policy
    
    def raw_parity(self,probs):

        policy = torch.zeros(self.n_actions)
        for i in range(len(probs)):
            a=[]
            for m in range(int(np.log2(self.n_actions))):
                if m==0:    
                    bitstring = np.binary_repr(i,width=self.n_qubits)
                else:
                    bitstring = np.binary_repr(i,width=self.n_qubits)[:-m]
                
                a.append(bitstring.count("1") % 2)
            policy[int("".join(str(x) for x in a),2)] += probs[i]

        policy = torch.tensor(policy)

        return policy 
    

    def softmax(self,probs,beta=1):
        
        if len(probs) == self.n_actions:
            scaled_output = probs * beta
            softmax_output = F.softmax(scaled_output, dim=0)
            softmax_output = torch.tensor(softmax_output)

            return softmax_output
        
        else:
            probs_flatten = probs.flatten()
            chunk_size = len(probs_flatten) // self.n_actions
            remainder = len(probs_flatten) % self.n_actions

            policy = [0] * self.n_actions

            for i in range(self.n_actions):

                start = i * chunk_size
                end = (i + 1) * chunk_size

                if i < remainder:
                    end += 1

                policy[i] = sum(probs_flatten[start:end])
                
            policy = torch.tensor(policy)

            scaled_output = policy / beta
            softmax_output = F.softmax(scaled_output, dim=0)
            softmax_output = torch.tensor(softmax_output)

            return softmax_output
