# Deep Q-Network (DQN) Algorithm

This repository contains an implementation of the **Deep Q-Network (DQN)** algorithm. DQN is a reinforcement learning algorithm that combines **Q-Learning** with **deep neural networks** and **experience replay**, enabling agents to learn optimal policies in complex environments.

## Algorithm Overview

### 1. **Core Concepts**
- **Q-Learning**: A value-based method that estimates the optimal action-value function $Q^*(s, a)$, which represents the expected reward for taking an action $a$ in a state $s$.
- **Deep Learning**: A neural network is used to approximate the $Q$-function, allowing it to handle high-dimensional input spaces, such as images or complex state vectors.
- **Experience Replay**: A buffer is used to store past experiences, which are replayed to train the network. This reduces correlation between training samples and improves stability.

### 2. **Key Components**
- **Replay Buffer**: Implemented in `ReplayBuffer`, it stores transitions $(s, a, r, s', \text{done})$. The `store_transition` method saves new experiences, and the `sample_buffer` method retrieves random minibatches for training.
- **Epsilon-Greedy Exploration**: Controlled in `select_action`, the agent chooses random actions with probability $\epsilon$ and selects greedy actions based on the $Q$-function otherwise. $\epsilon$ decays over time from `initial_exploration` to `final_exploration`.
- **Target Network**: A separate neural network (`q_target_network`) provides stable target values for training by being updated less frequently than the main `q_network`.

---

## DQN Algorithm Steps

1. **Initialization**:
   - Create two networks:
      - Main network (`q_network`): Initialized with random weights.
      - Target network (`q_target_network`): A copy of the main network.
   - Initialize a replay buffer (`ReplayBuffer`) to store transitions.
   - Set the exploration parameter $\epsilon$ to `initial_exploration`.

2. **Environment Interaction** (`train` method):
   - Reset the environment and start a new episode.
   - For each step:
      - Use `select_action` to choose an action $a_t$:
         - With probability $\epsilon$, select a random action.
         - Otherwise, select $\arg\max_a Q(s_t, a; \theta)$, where $\theta$ are the current network parameters.
      - Execute $a_t$ in the environment and observe the next state $s_{t+1}$, reward $r_t$, and termination flag.
      - Store the transition $(s_t, a_t, r_t, s_{t+1}, \text{done})$ in the replay buffer using `store_transition`.

3. **Training** (`update_model` method):
   - If the replay buffer has enough samples, sample a minibatch of transitions $(s_j, a_j, r_j, s'_{j+1}, \text{done}_j)$ using `sample_buffer`.
   - Compute the **target Q-value** $y_j$:
      - $y_j = r_j$ if the episode ends at $j+1$.
      - $y_j = r_j + \gamma \max_a Q'(s'_{j+1}, a; \theta^-)$, otherwise, where $Q'$ is the target network.
   - Compute the **current Q-value** $Q(s_j, a_j; \theta)$ using the main network.
   - Minimize the loss $\mathcal{L} = \frac{1}{N} \sum_j (y_j - Q(s_j, a_j; \theta))^2$ using the `compute_loss` method and update the network weights via backpropagation.

4. **Target Network Update**:
   - Periodically copy the weights of the main network to the target network (`q_target_network.load_state_dict`), controlled by the `target_update` parameter.

5. **Evaluation** (`evaluate` method):
   - At intervals, run a fixed number of episodes using only the greedy policy ($\epsilon = 0$).
   - Log the mean reward across these episodes.

---

## Implementation Details

### 1. **Hyperparameters**
- Discount factor ($\gamma$): `Config.gamma`
- Learning rate: `Config.lr`
- Replay buffer size: `Config.buffer_size`
- Minibatch size: `Config.minibatch_size`
- Initial exploration ($\epsilon$): `Config.initial_exploration`
- Final exploration: `Config.final_exploration`
- Target update frequency: `Config.target_update`
- Training steps: `Config.training_steps`

### 2. **Neural Network**
- Implemented in `SimpleMLP` (passed to `Config.q_network`).
- Configurable number of hidden layers and dimensions.

### 3. **Environment**
- The environment is created using the Gymnasium library. The default environment is `CartPole-v1`.

### 4. **Logging**
- Training and evaluation metrics are logged via `ExperienceLogger`, including loss, rewards, and exploration rates.

---

## References

- **Paper**: [Human-level control through deep reinforcement learning](dqn.pdf)

Feel free to explore the code to understand how these components are implemented in detail!
