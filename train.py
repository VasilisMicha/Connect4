import torch
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

env = ConnectFour()
# Get the number of state observations
state, _ = env.reset()
state = state.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(state.shape, COLUMNS, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE, TAU, LR, env, device)

while True:
    action = agent.select_action(state)
    stats = env.step(action.item())
    agent.store_transition(state, action, stats)
    state = stats[0].unsqueeze(0)
    if env.get_terminated():
        break
