# Deep Q-Learning Connect Four Agent

An implementation of a Deep Q-Network (DQN) reinforcement learning agent trained to master Connect Four ($6 \times 7$) through self-play and curriculum checkpointing, built with PyTorch and Gymnasium.

---

## Features

- **Convolutional Q-Network (`DQN`)**: Processes a $2 \times 6 \times 7$ spatial board representation (Channel 0: Agent tokens, Channel 1: Opponent tokens) using multi-layer 2D convolutions followed by fully connected layers with dropout regularization.
- **Action Masking**: Prevents invalid moves by dynamically masking saturated columns (Q-values set to $-\infty$) during both training, target network evaluation, and inference.
- **Self-Play & Historical Pool**: The agent plays against frozen checkpoints of its past iterations or random opponents, avoiding catastrophic forgetting and policy cycling.
- **Heuristic Exploration Fallbacks**: Exploratory steps incorporate one-step win/block threat detection to prioritize meaningful transitions in early training.
- **Soft Target Updates**: Employs Polyak averaging ($\tau = 0.005$) alongside replay memory sampling for stable temporal-difference (TD) updates.
- **Interactive CLI Interface**: Play directly against the latest trained checkpoint via `human_vs_robot.py` with colored terminal output.
- **Automated Logging**: Logs win rates, draw rates, average game durations, losses, and average max Q-values to CSV.

---

## Project Structure

```text
.
├── agent.py            # DQNAgent class, action selection, memory buffer & gradient step
├── connect_four.py     # Custom Gymnasium environment with rule logic & self-play pool
├── model.py            # PyTorch Convolutional DQN neural network architecture
├── train.py            # Training loop with curriculum progression & checkpointing
├── logger.py           # CSV metric logger for win rates, loss, and Q-values
├── human_vs_robot.py   # Interactive CLI interface to play against the trained agent
└── experiments/        # Auto-generated experiment runs, logs, and saved checkpoints (.pth)
```

---

## Neural Network Architecture

The Q-network (`model.py`) receives a tensor of shape `(batch_size, 2, 6, 7)`:

1. **Conv2D**: $2 \to 64$ filters, $3 \times 3$ kernel, padding = 1, ReLU
2. **Conv2D**: $64 \to 128$ filters, $3 \times 3$ kernel, padding = 1, ReLU
3. **Conv2D**: $128 \to 128$ filters, $3 \times 3$ kernel, padding = 1, ReLU
4. **Flatten**: $128 \times 6 \times 7 = 5376$
5. **Linear**: $5376 \to 256$, ReLU, Dropout ($p = 0.3$)
6. **Linear Output**: $256 \to 7$ (Q-values corresponding to columns $0 \dots 6$)

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support (optional, CPU fallback supported)

### Install Dependencies

```bash
pip install torch torchvision numpy gymnasium
```

---

## Training the Agent

To initiate training with default hyperparameters:

```bash
python train.py
```

### Training Highlights & Progression

- **Curriculum Milestone**: When the agent achieves $\ge 60\%$ win rate across 100 games against the latest benchmark opponent, a new model checkpoint (`model_weights_v{N}.pth`) is saved to `experiments/<run_name>-<timestamp>/`.
- **Opponent Rotation**: The newly saved model is appended to the pool of opponents (`stored_models`), driving iterative self-play improvement.
- **Evaluation**: Every 100 games, 10 deterministic test games are evaluated against a random opponent to gauge baseline competency.
- **Metrics**: Real-time stats are written to `training_stats.csv` inside the experiment directory.

### Key Hyperparameters

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `batch_size` | `128` | Minibatch size sampled from replay memory |
| `gamma` | `0.99` | Discount factor for future rewards |
| `eps_start` / `eps_end` | `0.9` / `0.01` | Starting and minimum exploration rate |
| `eps_decay` | `50000` | Exponential decay rate parameter |
| `replay_size` | `100000` | Maximum transitions in the replay buffer |
| `tau` | `0.005` | Soft update factor for target network |
| `lr` | `5e-4` | Learning rate for AdamW optimizer |
| `win_rate` | `0.60` | Win threshold over 100 games to promote and save model |

---

## Playing Against the Agent

Once at least one model checkpoint has been saved under `experiments/`, you can play an interactive game in your terminal:

```bash
python human_vs_robot.py
```

- You will be asked if you want to play first (`y/n`).
- Enter column numbers `0` through `6` to drop your token (`O` in blue).
- The agent plays as `X` (in red).
```
  0 1 2 3 4 5 6
 ---------------
| . . . . . . . |
| . . . . . . . |
| . . . X . . . |
| . . O X . . . |
| . O X O . . . |
| O X X O . . . |
 ---------------
```

---

## Logging & Monitoring

Training metrics are saved under `experiments/<experiment-name>-<timestamp>/training_stats.csv`:
- **Games Played**: Total completed episodes.
- **Win Rate against Models**: Rolling win rate over the past 100 matches vs checkpointed models.
- **Draw Rate against Models**: Rolling draw rate over the past 100 matches.
- **Win Rate against Random Moves**: Performance during evaluation rounds.
- **Average Game Length**: Mean steps per game.
- **Average Loss**: Huber (Smooth L1) loss per optimization step.
- **Average Max Q value**: Mean predicted value of chosen actions.