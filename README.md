# A Python Library for Analysis of VQC-based Quantum Policy Gradient Algorithms

This repository contains the implementation of a Python package developed as part of the dissertation titled **"Data Re-uploading and Expressivity/Trainability Trade-off on Quantum Policy Gradients."**

---

## Overview

This package is designed to integrate quantum computing with reinforcement learning, specifically focusing on using **Variational Quantum Circuits (VQCs)** as function approximators in **Quantum Reinforcement Learning (QRL)**, with emphasis on **Policy Gradient Methods**. It facilitates the exploration and evaluation of data re-uploading strategies, examining their impact on the expressivity, trainability, and overall performance of quantum-based reinforcement learning models.

- Pre-built VQC architectures.
- Flexible configuration files for defining hyperparameters.
- Support for running parallel quantum agents.
- Comprehensive analysis tools for trainability and performance.

### Features

1. **Integrated Quantum Circuit Architectures**: Implements the circuits in [Jerbi et al. (2021)](https://arxiv.org/abs/2103.05577), [TensorFlow Quantum tutorial (Broughton et al., 2021)](https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning), and [Universal Quantum Classifier (Pérez-Salinas et al., 2020)](https://arxiv.org/abs/1906.10594) adapted to RL.
2. **Policy Options**: Different policy post-processing strategies, including the Born policies developed in [Jerbi et al. (2021)](https://arxiv.org/abs/2103.05577) and [Meyer et al. (2023)](https://arxiv.org/abs/2212.06663), as well as the Softmax policy, which can implement beta scheduling to adjust the agents' greediness.
3. **Extensible Configurations**: Big flexibility of parameters, allowing the use of input and output scaling, different entangling patterns, full liberty of the observables used, and more.
4. **Parallel Execution**: Train multiple agents concurrently using CLI arguments.
5. **Experimentation Framework**: Allows benchmark environments from OpenAI, such as **CartPole-v1** and **Acrobot-v1**.

### Technologies

- **Core Programming Language**: [Python (Recommended Version: 3.11.0](https://www.python.org/)
- **Machine Learning**: [PyTorch](https://pytorch.org/)
- **Quantum Computing**: [PennyLane](https://pennylane.ai/)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vqc_policygradients.git
   cd vqc_policygradients
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage


1. **Notebook Exploration**: The [Jupyter Notebook](./notebook.ipynb) offers an interactive way to explore the package's functions, making it ideal for first-time users. It also supports agent training, though this method is typically more suited for experimentation and exploration.
2. **Command-Line Interface**:
   - To train agents, run [main.py](./main.py) from the command line, specifying:
     - Path to a configuration file.
     - Number of agents to train in parallel.
       
```bash
python main.py <config_path> <num_agents>
```

We provide several configuration files in `configs/`. Users can also create custom configurations for experiments and load them via CLI.

---

## Directory Structure

```
📦vqc-policy-gradients
 ┣ 📂configs               # Configuration files for experiments (pre-defined and customizable)
 ┣ 📂vqcpg                 # Package directory
 ┃ ┣ 📂agent               # REINFORCE agent class for training
 ┃ ┣ 📂analysis            # Plotting and analysis tools
 ┃ ┣ 📂model               # Quantum circuits, operations, and policies
 ┃ ┗ 📂utils               # Helper functions
 ┣ 📜LICENSE.md            # License
 ┣ 📜README.md             # Project documentation
 ┣ 📜main.py               # Entry point for training agents
 ┣ 📜notebook.ipynb        # Jupyter notebook for exploration
 ┗ 📜requirements.txt      # Required libraries and versions
```

---

## Contact

For questions or feedback, contact the author at:
- **Email**: soaresberna[at]gmail[dot]com

---

## License

This project is licensed under the [MIT License](LICENSE.md). 

When utilizing or modifying this library, please include a copy of the MIT License and give appropriate credit to the original authors.

---
