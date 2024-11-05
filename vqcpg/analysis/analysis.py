import pandas as pd
import numpy as np
import os

class Analysis:
    def __init__(self, path_to_dir):
        self.path = path_to_dir
        separated_data = {
            'episode_reward': [],
            'loss': [],
            'runtime': [],
            'params_gradients': []
        }

        for subdir, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(subdir, file)
                    if os.path.exists(file_path):
                        data = np.load(file_path, allow_pickle=True)
                        for key in separated_data:
                            if key in data:
                                separated_data[key].append(data[key].tolist())
        self.data = separated_data

    def get_rewards(self):
        return self.data["episode_reward"]

    def get_loss(self):
        return self.data["loss"]

    def get_runtime(self):
        return self.data["runtime"]

    def get_gradients(self):
        return self.data["params_gradients"]

    def get_moving_average(self, window_size=10):
        rewards = self.get_rewards()
        moving_averages = []
        for reward in rewards:
            moving_averages.append(pd.Series(reward).rolling(window_size).mean())
        return moving_averages

    def compute_mean_loss(self):
        # Get loss values
        losses = self.get_loss()

        # Determine the minimum number of episodes across all agents
        min_length = min(len(losses[i]) for i in range(len(losses)))

        # Trim each agent's loss values to the minimum length
        losses = [losses[i][:min_length] for i in range(len(losses))]

        # Convert losses to a numpy array with shape (n_agents, n_episodes)
        losses_array = np.array([np.squeeze(agent_losses) for agent_losses in losses])

        # Compute mean loss per episode across all agents
        mean_loss_per_episode = np.mean(losses_array, axis=0)

        return mean_loss_per_episode
    
    def compute_norm_and_variance(self):

        gradients = self.get_gradients()
        min_length = min([len(gradients[i]) for i in range(len(gradients))])

        gradients = [gradients[i][:min_length] for i in range(len(gradients))]

        def flatten_gradients(gradients):
            for i in range(len(gradients)):
                for j in range(len(gradients[i])):
                    gradients[i][j] = np.concatenate([lista.flatten() for lista in gradients[i][j]], axis = 0)

        flatten_gradients(gradients)

        gradients_array = np.array(gradients)

        magnitudes_gradients = np.linalg.norm(gradients_array, axis = 2)

        mean_magnitudes_gradients = np.insert(np.mean(magnitudes_gradients, axis = 0),0,0)

        std_magnitudes_gradients  = np.insert(np.var(magnitudes_gradients, axis = 0),0,0)

        return mean_magnitudes_gradients, std_magnitudes_gradients