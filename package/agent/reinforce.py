import torch
import os
import json
import numpy as np
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from utils import tensor_to_list


class ReinforceUpdate():

    def __init__(self, pqc, optimizer, env, env_name, n_episodes, max_t, gamma, print_every, verbose, file_name = None):
        
        self.pqc = pqc
        self.optimizer = optimizer
        self.env = env
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.gamma = gamma
        self.scores_deque = deque(maxlen=print_every)
        self.print_every = print_every
        self.verbose = verbose
        self.file_name = file_name
        self.running_reward = 10

    def get_trajectory(self):

        self.saved_log_probs = []
        self.rewards = []
        state = self.env.reset()
        for t in range(self.max_t):
            if t == 0:
                state_tensor = torch.tensor(state[0]).float()
            else:
                state_tensor = torch.tensor(state).float()
            action, log_prob, _, = self.pqc.sample(state_tensor)
            state, reward, done, _, _ = self.env.step(action)
            
            self.saved_log_probs.append(log_prob)
            self.rewards.append(reward)

            if done:
                self.scores_deque.append(sum(self.rewards))
                break

    def update_policy(self):

        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for log_prob, ret in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * ret)

        policy_unsqueezed = [torch.unsqueeze(loss, 0) for loss in policy_loss]
        self.loss = torch.cat(policy_unsqueezed).sum()

        self.optimizer.zero_grad()
        self.loss.backward()    
        self.optimizer.step()
        
    def save_agent_data(self,path):

        agent_variables = {
            "Number of Qubits": self.pqc.circuit.n_qubits,
            "Number of Layers": self.pqc.circuit.n_layers,
            "Shots": self.pqc.circuit.shots,
            "Input Scaling": self.pqc.circuit.input_scaling,
            "Design": self.pqc.circuit.design,
            "Differentiation Method": self.pqc.circuit.diff_method,
            "Weight Initiation": str(self.pqc.circuit.weight_init),
            "Input_init": str(self.pqc.circuit.input_init),
            "Measure": str(self.pqc.circuit.measure),
            "Measure Qubits": self.pqc.circuit.measure_qubits,
            "Policy Type": self.pqc.policy.post_processing,
            "Optimizers": str(self.optimizer),
            "Envinronment Name": str(self.env_name),
            "Gamma (discounting factor)": self.gamma,
        }

        with open(os.path.join(path, "agent_characteristics.json"), "w") as f:
            json.dump(agent_variables, f, indent=4)

    def save_gradients(self,path):

        short_gradient1 = [round(float(value), 16) for value in tensor_to_list(self.pqc.get_gradients()[0])]
        short_gradient2 = [round(float(value), 16) for value in tensor_to_list(self.pqc.get_gradients()[1])]

        gradients = [short_gradient1, short_gradient2]
        new_path = os.path.join(path, "gradients_data_" + str(self.file_name) + ".json")
        with open(new_path, 'a') as f:

            json.dump([gradients], f, indent=4)

    def writer_function(self, writer1, iteration):

        writer1.add_scalar("Episode Reward", np.mean(self.scores_deque), global_step=iteration)
        writer1.add_scalar("Running Reward", self.running_reward, global_step=iteration)
        writer1.add_scalar("Runtime", self.runtime, global_step=iteration)
        writer1.add_scalar("Loss", self.loss.item(), global_step=iteration)
                    
    def train(self):
        
        logs_dir = "data"
        os.makedirs(logs_dir, exist_ok=True)
        envinronment_folder = os.path.join(logs_dir, self.env_name)
        experiment_folder = f"{self.pqc.policy.post_processing}_{self.pqc.circuit.n_layers}layer"
        experiment_path = os.path.join(envinronment_folder, experiment_folder)
        os.makedirs(experiment_path, exist_ok=True)
        run = os.path.join(experiment_path,str(self.file_name))
        os.makedirs(run, exist_ok=True)
        writer = SummaryWriter(log_dir=run)
        self.save_agent_data(experiment_path)
        
        for i in range(1, self.n_episodes):
            start_time = time.time()
            self.get_trajectory()
            self.pqc.T_schedule(i,self.n_episodes)
            self.update_policy()
            end_time = time.time()
            self.runtime = end_time - start_time
            self.writer_function(writer,i)
            self.save_gradients(experiment_path)
            self.running_reward = (self.running_reward * 0.99) + (len(self.rewards) * 0.01)
            
            if self.running_reward > self.env.spec.reward_threshold:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i, np.mean(self.scores_deque)))
                break
            elif i % self.print_every == 0 and self.verbose == 1:
                print('Episode {}\tLast reward: {:.2f}\tLast {} Episodes average reward: {:.2f}\tRuntime: {:.2f}\t {:.2f}\t'.format(i, self.scores_deque[-1], self.print_every, np.mean(self.scores_deque), self.runtime, self.running_reward))