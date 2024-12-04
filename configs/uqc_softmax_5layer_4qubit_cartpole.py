import pennylane as qml
import torch
import numpy as np
from functools import partial
import os

from vqcpg.model.vqc_designs import UQC
from vqcpg.model.vqc_observables import two_measure_expval


n_qubits = 4
n_layers = 5
state_dim = 4
device = 'lightning.qubit'
shots = None
diff_method = 'adjoint' 
encoding = 'full'
entanglement = True
entanglement_pattern = "all_to_all"
entanglement_gate = qml.CZ
input_init = partial(torch.nn.init.normal_, mean=0.0, std=0.01)
weight_init = lambda shape, dtype=torch.float: torch.FloatTensor(shape).uniform_(-np.pi, np.pi)
bias_init = torch.nn.init.zeros_
policy_circuit_measure = two_measure_expval
policy_circuit = UQC(n_qubits,
                    n_layers, 
                    state_dim,
                    device,
                    shots, 
                    diff_method,
                    encoding,
                    entanglement, 
                    entanglement_pattern, 
                    entanglement_gate,
                    input_init,
                    weight_init,
                    bias_init, 
                    policy_circuit_measure)

#   Post processing settings
n_actions = 2
post_processing = 'softmax'
beta_scheduling = False
beta = 1
increase_rate = 0.003
output_scaling = True
output_init = torch.nn.init.ones_

policy_lr_list= [0.1, 0.01, 0.1, 0.1]  # [weights, params, bias, output_weights]

env_name = 'CartPole-v1'
n_episodes = 1000
max_t = 500
gamma = 0.98
baseline = True
batch_size = 10
normalize = True
print_every = 100
verbose = 1

data_path = os.getcwd()
tensorboard = True
