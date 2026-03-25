import torch
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
from collections import deque
from itertools import count
from connect_four import ConnectFour
from agent import DQNAgent
from logger import Logger
from model import DQN

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
win_history = deque(maxlen=100)
win_rate = 0
games_played = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ConnectFour()
logger = Logger()

agent = DQNAgent(COLUMNS, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, REPLAY_SIZE, TAU, LR, env, device)

while True:
    while len(win_history) != 100 or win_rate < 0.60:
        state, _ = env.reset()
        state = state.unsqueeze(0).to(device)
        episode_reward = 0
        sum_loss = 0
        sum_max_q = 0
        t = 0
        for t in count():
            action = agent.select_action(state).to(device)
            stats = env.step(action.item())
            agent.store_transition(state, action, stats)
            state = stats[0].unsqueeze(0).to(device)

            episode_reward += stats[1]

            loss, max_q = agent.optimize()
            if loss is not None and max_q is not None:
                sum_loss += loss
                sum_max_q += max_q

            agent.update_target_network()

            terminated = stats[2]
            if terminated:
                episode_durations.append(t + 1)
                break

        episode_rewards.append(episode_reward)
        if env.get_opponent() and episode_reward != 0:
            win_history.append(1 if episode_reward > 0 else 0)
            win_rate = sum(win_history) / len(win_history)

        logger.log_training(episode_reward, env.get_opponent(), t+1, sum_loss/(t+1), sum_max_q/(t+1))

        games_played += 1
        print(f"{games_played}) {win_rate:.2f}", end="-")


    win_history = deque(maxlen=100)
    win_rate = 0
    models_dir = project_root / "models"
    model_files = os.listdir(models_dir)
    models_num = len(model_files)
    torch.save(agent.get_policy_net().state_dict(), project_root / "models" / f"model_weights_v{models_num+1}.pth")
    print(f"----------------Saving model_weights_v{models_num+1}.pth-------------------------")

    new_version = DQN(actions=COLUMNS).to(device)
    new_version.load_state_dict(agent.get_policy_net().state_dict())
    new_version.eval()

    env.stored_models.append(new_version)
