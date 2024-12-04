import pennylane as qml
import torch
import numpy as np
from functools import partial
import os

from vqcpg.model.vqc_designs import TfqTutorial
from vqcpg.model.vqc_observables import two_measure_expval


n_qubits = 4
n_layers = 5
device = 'lightning.qubit'
shots = None
diff_method = 'adjoint' 
entanglement = True
entanglement_pattern = 'all_to_all'
entanglement_gate = qml.CZ
input_scaling = True
input_init = partial(torch.nn.init.normal_, mean=0.0, std=0.01)
weight_init = lambda shape, dtype=torch.float: torch.FloatTensor(shape).normal_(-np.pi, np.pi)
policy_circuit_measure = two_measure_expval
policy_circuit = TfqTutorial(n_qubits, 
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
                            policy_circuit_measure)


n_actions = 2
post_processing = 'softmax'
beta_scheduling = False
beta = 1
increase_rate = 0.0005
output_scaling = True
output_init = torch.nn.init.ones_

policy_lr_list= [0.1, 0.01, 0.1]  # [input_weights, weights, output_weights]

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
