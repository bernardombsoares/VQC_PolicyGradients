import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gym

import os
import json
import time
import warnings
from collections import deque

from vqcpg.utils.utils import create_directory


class ReinforceAgent:
    '''
    For detailed information about the parameters, call the info() method.
    '''
    def __init__(self, 
                policy, 
                policy_optimizer, 
                env_name, 
                n_episodes, 
                max_t, 
                gamma, 
                baseline, 
                batch_size, 
                normalize,
                print_every, 
                verbose):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.policy_optimizer = policy_optimizer
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.gamma = gamma
        self.baseline = baseline
        self.batch_size = batch_size
        self.normalize = normalize
        self.print_every = print_every
        self.verbose = verbose

        self.solved = False
        self.scores = deque(maxlen=100)
        self.batch_log_probs = []
        self.batch_rewards = []
        self.loss = torch.tensor(0.0)

    def train(self, run_count=None, rundate = None, path=None, tensorboard=False):

        # Creates data saving files if specified
        if run_count is not None and path is not None:
            data_path = create_directory(os.path.join(path, 'data'))
            env_folder = create_directory(os.path.join(data_path, self.env_name))
            experiment_folder_name = f"{self.policy.circuit.__class__.__name__}_{self.policy.circuit.n_qubits}qubits_{self.policy.circuit.n_layers}layer_{rundate}"
            experiment_path = create_directory(os.path.join(env_folder, experiment_folder_name))
            run_path = create_directory(os.path.join(experiment_path, f'run_{str(run_count)}'))
            self.save_agent_data(experiment_path)
        

        # Create TensorBoard session if specified
        if tensorboard:
            writer = SummaryWriter(log_dir=run_path)

        for i in range(1, self.n_episodes + 1):
            start_time = time.time()

            # Get episode
            self.get_trajectory()

            # Check if environment is solved
            self.env_solved_verification()

            # Update parameters if the batch is full and environment is not solved
            if i > 1 and i % self.batch_size == 0 and not self.solved:
                self.update_policy()
                self.policy.post_processing.beta_schedule()
 
            end_time = time.time()

            # Calculate the runtime
            self.runtime = end_time - start_time

            # Write in the TensorBoard session
            if tensorboard:
                self.writer_function(writer, i)

            # Save the episode data
            if run_count is not None and path is not None:
                self.save_data(run_path, i)

            # Print out episode data
            if self.verbose == 1:
                print('Episode {} reward: {:.2f}\t Solved: {}'.format(i, self.scores[-1], self.solved))
            if self.verbose >= 2:
                print('Episode {} reward: {:.2f}\t Solved: {}\t Runtime: {:.2f}\t Loss: {:.2f}'.format(i, self.scores[-1], self.solved, self.runtime, self.loss))
            if i % self.print_every == 0 and i > 1:
                print('Last {} Episodes average reward: {:.2f}\t'.format(len(self.scores), np.mean(self.scores)))

        # Save the final weights
        self.save_final_weights()

        # Close TensorBoard session
        if tensorboard:
            writer.close()

    def get_trajectory(self):
        '''
        Gets a trajectory based on the running policy until it runs out of bounds or achieves maximum reward of an episode
        '''
        # Get an episode trajectory
        self.saved_log_probs = []
        self.rewards = []
        state = self.env.reset()[0]
        for t in range(self.max_t):
            state_tensor = torch.tensor(self.normalize_state(state)).float().to(self.device)
            action, log_prob, _ = self.policy.sample(state_tensor)
            state, reward, done, _, _ = self.env.step(action)
            
            self.saved_log_probs.append(log_prob)
            self.rewards.append(reward)

            if done:
                break

        # Save the episode reward
        self.scores.append(sum(self.rewards))

        # Save data from the episode to the batch
        self.batch_log_probs.append(self.saved_log_probs)
        self.batch_rewards.append(self.rewards)

        # Clear data in case the agent already solved the environment
        if self.solved is True:
            self.batch_log_probs = []
            self.batch_rewards = []
      
    def update_policy(self):
        '''
        Computes the loss and gradients and updates the policy via gradient methods
        '''
        # Discounting of the rewards
        all_returns = []
        for batch in self.batch_rewards:
            R = 0
            ep_return = []
            for r in reversed(batch):
                R = r + self.gamma * R
                ep_return.insert(0, R)
            ep_return = torch.tensor(ep_return).to(self.device)

            # Standardization of the discounted returns
            ep_return = (ep_return - ep_return.mean()) / (ep_return.std() + 1e-8)

            all_returns.append(ep_return)

        # Calculate the policy loss
        policy_loss = []     
        if self.baseline:
            baseline = np.mean([sum(lst) for lst in all_returns])
            for log_probs, ep_returns in zip(self.batch_log_probs, all_returns):
                for log_prob, ret in zip(log_probs, ep_returns):
                    advantage = ret - baseline 
                    policy_loss.append(-log_prob * advantage)
        else:
            for log_probs, ep_returns in zip(self.batch_log_probs, all_returns):
                for log_prob, ret in zip(log_probs, ep_returns):
                    policy_loss.append(-log_prob * ret)

        policy_unsqueezed = [torch.unsqueeze(loss, 0) for loss in policy_loss]
        self.loss = torch.cat(policy_unsqueezed).mean()

        # Compute the gradients 
        self.policy_optimizer.zero_grad()
        self.loss.backward()
        self.policy_optimizer.step()

        # Clear old data
        del all_returns
        del policy_loss
        del policy_unsqueezed 
        self.batch_log_probs = []
        self.batch_rewards = []

    def normalize_state(self, state):
        '''
        Processes the input state by reducing its dimensionality and normalizing it
        '''
        # State-space reduction for the Acrobot
        if self.env_name in ('Acrobot-v0', 'Acrobot-v1'):
            theta1 = np.arccos(state[0])
            theta2 = np.arccos(state[2])
            state = [theta1,theta2,state[4],state[5]]


        # Normalize each feature by the maximum absolute value at each step
        if self.normalize == True:
            max_abs_value = max(abs(value) for value in state)
            state = np.array([value / max_abs_value for value in state])
        
        return state
    
    def env_solved_verification(self):
        '''
        Checks if the environment is solved
        '''
        # Acrobot-v1
        if self.env_name in ('Acrobot-v1'):
            if np.mean(self.scores) > -125:
                self.solved = True
        
        # CartPole-v0 and CartPole-v1
        elif self.env_name in ('CartPole-v0','CartPole-v1'):
            if np.mean(self.scores) > self.env.spec.reward_threshold:
                self.solved = True
        
        else:              
            warnings.warn(f"No reward threshold defined for environment {self.env_name}. "
                          "Consider specifying a solved condition explicitly.",
                          UserWarning
            )

    def save_agent_data(self, main_path):
        '''
        Stores the most relevant model parameters into a .json file.
        '''
        # Use the get_parameters method to get Circuit Parameters, Policy Parameters and Agent Parameters
        circuit_params = self.policy.circuit.get_parameters()  # Get circuit parameters
        policy_params = self.policy.post_processing.get_parameters()  # Get policy parameters
        agent_params = self.get_parameters()  # Get agent parameters
        
        # Create a structured dictionary
        agent_variables = {
            "Circuit Parameters": circuit_params,
            "Policy Parameters": policy_params,
            "Agent Parameters": agent_params
        }

        # Convert sets to lists
        def convert_sets_to_lists(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {key: convert_sets_to_lists(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            else:
                return obj

        # Convert sets in agent_variables
        agent_variables = convert_sets_to_lists(agent_variables)

        # Save as JSON
        with open(os.path.join(main_path, "agent_characteristics.json"), "w") as f:
            json.dump(agent_variables, f, indent=4)

    def save_data(self, run_path, iteration):
        '''
        Saves the data into a .npz file for each episode
        '''
        data_file = os.path.join(run_path, "run_data.npz")
        
        # Load existing data if the file exists
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            old_episode_reward = data['episode_reward'].tolist()
            old_loss = data['loss'].tolist()
            old_runtime = data['runtime'].tolist()
            old_gradients = data['gradients'].tolist()
            old_params = data['params'].tolist()
        else:
            old_episode_reward = []
            old_loss = []
            old_runtime = []
            old_gradients = []
            old_params = []

        # Add episode reward and runtime
        old_episode_reward.append(self.scores[-1])
        old_runtime.append(self.runtime)

        # Stores the loss and parameter gradients when batch is full
        current_episode_gradients = []
        current_episode_params = []
        if iteration % self.batch_size == 0 and iteration > 1 and self.solved is False:
            old_loss.append(self.loss.item())
            for name, param in self.policy.circuit.named_parameters():
                # Get the parameter values and flatten them
                param_array = param.cpu().detach().numpy().flatten()
                
                # Append the parameter values to the old_parameters list
                current_episode_params.append(param_array)

                # If the parameter has a gradient, get and flatten it
                if param.grad is not None:
                    grad_array = param.grad.cpu().numpy().flatten()
                    current_episode_gradients.append(grad_array)

            # Concatenate the parameters and gradients into a single array for each episode
            flattened_parameters = np.concatenate(current_episode_params)
            flattened_gradients = np.concatenate(current_episode_gradients)

            old_params.append(flattened_parameters)
            old_gradients.append(flattened_gradients)
            
        # Save data to .npz file
        np.savez_compressed(data_file,
                            episode_reward=np.array(old_episode_reward),
                            loss=np.array(old_loss),
                            runtime=np.array(old_runtime),
                            gradients=np.array(old_gradients, dtype=object),
                            params=np.array(old_params,dtype=object))

        # Clear old data lists to free up memory
        del old_episode_reward[:]
        del old_loss[:]
        del old_runtime[:]
        del old_gradients[:]
        del old_params[:]

    def save_final_weights(self, run_path):
        '''
        Saves the final model weights into the .npz file at the end of training.
        '''
        data_file = os.path.join(run_path, "run_data.npz")

        # Load existing data if the file exists
        if os.path.exists(data_file):
            data = dict(np.load(data_file, allow_pickle=True))
        else:
            data = {}

        # Extract and save final weights
        final_weights = {name: param.detach().cpu().numpy() for name, param in self.policy.named_parameters()}
        data['final_weights'] = final_weights

        # Save updated data with final weights to .npz file
        np.savez_compressed(data_file, **data)
    
    def writer_function(self, writer, iteration):
        '''
        Stores data into a tensorboard session
        '''
        writer.add_scalar("Episode Reward", self.scores[-1], global_step=iteration)
        writer.add_scalar("Runtime", self.runtime, global_step=iteration)
        writer.add_scalar("Loss", self.loss.item(), global_step=iteration)

        gradients = []
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                if name == 'input_params' or name == 'params':
                    gradients.append(param.grad.view(-1))

        # Concatenate all collected gradients into a single tensor and calculate L2 norm of the combined gradients
        if gradients:
            combined_gradients = torch.cat(gradients)
            combined_grad_norm = torch.norm(combined_gradients).item()
            
            # Log the combined gradient norm
            writer.add_scalar("Gradient Norm/Combined", combined_grad_norm, global_step=iteration)

    def get_parameters(self):
        # Extract specified attributes for JSON serialization
        return {
            "Environment": self.env_name,
            "Gamma (discounting factor)": self.gamma,
            "Baseline": self.baseline,
            "Batch Size": self.batch_size,
            "Normalize": self.normalize,
        }
    
    @classmethod
    def info(cls):
        '''
        Provides a summary of the ReinforceAgent class and its parameters/methods.
        '''
        info_text = """

        Implements a Reinforcement Learning agent using the REINFORCE algorithm.

        Parameters:
        ----------
        policy (PolicyType): 
            An instance of the policy class that defines the action selection process based on the outputs of the VQC.
        
        policy_optimizer (torch.optim.Optimizer): 
            Optimizer used for updating the policy parameters during training.
        
        env_name (str): 
            Name of the environment to interact with, typically defined in OpenAI Gym.
        
        n_episodes (int): 
            Total number of episodes for training the agent.
        
        max_t (int): 
            Maximum number of time steps per episode.
        
        gamma (float): 
            Discount factor for future rewards, where 0 < gamma < 1.
        
        baseline (bool): 
            If True, applies a baseline to reduce variance in the policy gradient estimates.
        
        batch_size (int): 
            Number of episodes after which the policy parameters will be updated.
        
        normalize (bool): 
            If True, normalizes the state input for the agent.
        
        print_every (int): 
            Number of episodes after which to print the average reward.
        
        verbose (int): 
            Level of verbosity for outputting training details (0: none, 1: basic, 2: detailed).

        Methods:
        -------
        train(self, run_count=None, rundate=None, path=None, tensorboard=False):
            Trains the agent by collecting episodes and updating the policy using the REINFORCE algorithm.

        get_trajectory(self):
            Gathers a trajectory from the environment using the current policy until the episode ends.

        update_policy(self):
            Computes the loss, applies gradients, and updates the policy parameters.

        normalize_state(self, state):
            Normalizes and processes the input state to reduce dimensionality.

        env_solved_verification(self):
            Checks if the environment has been solved based on the average score.

        save_agent_data(self, main_path):
            Saves the agent's parameters and model characteristics to a JSON file.

        save_data(self, run_path, iteration):
            Saves episode data, including rewards, loss, and gradients, to a .npz file.

        save_final_weights(self, run_path):
            Saves the final model weights at the end of training to a .npz file.

        writer_function(self, writer, iteration):
            Logs episode data to TensorBoard for visualization.

        get_parameters(self):
            Returns a dictionary of important agent parameters for JSON serialization.
        """
        return info_text