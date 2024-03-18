import numpy as np
import pennylane as qml
import torch

class PolicyType():
    
    def __init__(self, n_actions, n_qubits, policy_type = 'raw_contiguous'):
        self.policy_type = policy_type
        self.n_actions = n_actions
        self.n_qubits = n_qubits

    def raw_contiguous(self,probs):

        return probs
    
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

        return policy 
    

    def 