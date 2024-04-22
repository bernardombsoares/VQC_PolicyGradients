import sys
import torch
from joblib import Parallel, delayed

if __name__ == "__main__":

    path_to_file = sys.argv[1]

    sys.path.append(path_to_file.rsplit('/', 1)[0])

    import_name = path_to_file.rsplit('/', 1)[1][:-3]

    def train_agents(agent_number):
         
        from model.policy_structure import PolicyType
        from model.circuit_generator import CircuitGenerator
        from model.quantum_model import QuantumPolicyModel
        from agent.reinforce import ReinforceUpdate

        script = __import__(import_name)

        circuit = CircuitGenerator( script.n_qubits, script.n_layers,
                                    script.shots, script.input_scaling, 
                                    script.diff_method, script.weight_init,
                                    script.input_init, script.measure)
        
        policy = PolicyType(script.n_qubits, script.n_actions, script.post_processing)

        pqc = QuantumPolicyModel(circuit,policy)

        optimizer = torch.optim.Adam([{'params': params, 'lr': lr} for params, lr in zip(circuit.parameters(), script.lr_list)])
        
        reinforce_update = ReinforceUpdate(pqc, optimizer, 
                                           script.env, script.env_name, 
                                           script.n_episodes, script.max_t, 
                                           script.gamma, script.print_every, 
                                           script.verbose, agent_number)
        reinforce_update.train()

    
    num_agents = 10

    results = Parallel(n_jobs=num_agents)(delayed(train_agents)(i) for i in range(num_agents))