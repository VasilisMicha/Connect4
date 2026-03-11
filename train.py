import torch
from itertools import count
from connect_four import ConnectFour
from agent import DQNAgent

COLUMNS = 7

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 20000
REPLAY_SIZE = 100000
TAU = 0.005
LR = 1e-4

episode_durations = []
episode_rewards = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ConnectFour()

agent = DQNAgent(COLUMNS, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE, TAU, LR, env, device)

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = state.unsqueeze(0)
    episode_reward = 0
    for t in count():
        action = agent.select_action(state)
        stats = env.step(action.item())
        agent.store_transition(state, action, stats)
        state = stats[0].unsqueeze(0)

        episode_reward += stats[1]

        agent.optimize()
        agent.update_target_network()

        terminated = stats[2]
        if terminated:
            episode_durations.append(t + 1)
            break

    episode_rewards.append(episode_reward)
