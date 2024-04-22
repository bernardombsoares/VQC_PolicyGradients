import gym
import torch
import numpy as np

from pqc.measures import Measure


n_qubits = 4
n_layers = 4      #set to 1 if data_reuploading is off
n_actions = 2
shots = None
input_scaling = True
design = 'jerbi_circuit' 
diff_method = 'adjoint' 
weight_init = lambda shape, dtype=torch.float: torch.FloatTensor(shape).uniform_(-np.pi, np.pi)
input_init = torch.nn.init.ones_
measure_qubits = n_qubits
measure = Measure(measure_qubits).measure_probs()

post_processing = 'raw_contiguous'

lr_list= [0.01,0.08]
env_name = 'CartPole-v1'
env = gym.make(env_name)
n_episodes = 500
max_t = 500
gamma = 0.98
print_every = 10
verbose = 1