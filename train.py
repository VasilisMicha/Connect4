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
state, info = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(state.shape)

# policy_net = DQN(actions=COLUMNS).to(device)
# target_net = DQN(actions=COLUMNS).to(device)
# target_net.load_state_dict(policy_net.state_dict())
#
# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory(REPLAY_SIZE)


agent = DQNAgent(state.shape, COLUMNS, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE, TAU, LR, env, device)

while True:
    a = agent.select_action(state)
    print(f"action: {a}")
    action = a
    env.step(action)
    if env.get_terminated():
        break
