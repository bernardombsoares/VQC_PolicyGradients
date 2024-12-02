import os
import sys
import torch
import ray
from datetime import datetime
from vqcpg.model.policy import PolicyPostProcessing
from vqcpg.model.quantum_policy import QuantumPolicy
from vqcpg.agent.reinforce import ReinforceAgent


@ray.remote
def train_agents(script_module_name, agent_index, rundate):
    # Dynamically import the configuration script

    script = __import__(f"configs.{script_module_name}", fromlist=[''])

    # Set up the policy circuit
    policy_circuit = script.policy_circuit
    
    # Set up the policy post-processing
    policy_post_processing = PolicyPostProcessing(
        script.n_qubits, 
        script.n_actions, 
        script.post_processing, 
        script.beta_scheduling, 
        script.beta, 
        script.increase_rate, 
        script.output_scaling, 
        script.output_init
    )

    # Define the quantum policy and optimizer
    policy = QuantumPolicy(policy_circuit, policy_post_processing)
    policy_params = list(policy_circuit.parameters()) + list(policy_post_processing.parameters())
    policy_optimizer = torch.optim.Adam(
        [{'params': p, 'lr': lr} for p, lr in zip(policy_params, script.policy_lr_list)], 
        amsgrad=True
    )

    # Define the REINFORCE agent
    reinforce_update = ReinforceAgent(
        policy, 
        policy_optimizer, 
        script.env_name, 
        script.n_episodes, 
        script.max_t, 
        script.gamma, 
        script.baseline, 
        script.batch_size,
        script.normalize,
        script.print_every, 
        script.verbose
    )

    # Train the agent
    file_name = f"agent_{agent_index}"
    reinforce_update.train(file_name, rundate, script.data_path, script.tensorboard)
    
    return f"Agent {agent_index} completed training with success status: {reinforce_update.solved}"

if __name__ == "__main__":

    ray.init()

    path_to_file = sys.argv[1]
    config_dir = os.path.abspath(os.path.dirname(path_to_file))
    sys.path.append(config_dir)  # Add configs folder to the Python module search path
    import_name = os.path.splitext(os.path.basename(path_to_file))[0]
    full_path = os.path.join(config_dir, import_name + ".py")

    num_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rundate = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    all_results = []
    results = [train_agents.remote(import_name, i, rundate) for i in range(num_agents)]
    completed_results = ray.get(results)
    all_results.extend(completed_results)

    for idx, result in enumerate(completed_results):
        print(f"Result for agent {idx}: {result}")

    ray.shutdown()
