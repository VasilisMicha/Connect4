import torch
from datetime import datetime
import os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
from collections import deque
from itertools import count
from connect_four import ConnectFour
from agent import DQNAgent
from logger import Logger
from model import DQN


class DQNTrainer:
    def __init__(self,  columns=7, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=20000, replay_size=100000, tau=0.005, lr=1e-4, win_rate=0.6, experiment_name="experiment"):
        self.columns = columns
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replay_size = replay_size
        self.tau = tau
        self.lr = lr
        self.win_rate = win_rate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = project_root /  "experiments" / f"{experiment_name}-{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        episode_durations = []
        episode_rewards = []
        win_history = deque(maxlen=100)
        win_rate = 0
        games_played = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = ConnectFour()
        logger = Logger(self.exp_dir)

        agent = DQNAgent(self.columns, self.batch_size, self.gamma, self.eps_start, self.eps_end, self.eps_decay, self.replay_size, self.tau, self.lr, env, device)

        while True:
            while len(win_history) != 100 or win_rate < self.win_rate:
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

                        is_benchmark = (env.get_opponent() == env.stored_models[-1]) if env.stored_models else (env.get_opponent() is None)

                        if is_benchmark:
                            final_step_reward = stats[1]
                            if final_step_reward >= 1.0:   # True Win
                                win_history.append(1)
                            elif final_step_reward <= -1.0: # True Loss
                                win_history.append(0)

                        break

                episode_rewards.append(episode_reward)

                if len(win_history) > 0:
                    win_rate = sum(win_history) / len(win_history)

                logger.log_training(episode_reward, env.get_opponent(), t+1, sum_loss/(t+1), sum_max_q/(t+1))

                games_played += 1
                print(f"{games_played}) {win_rate:.2f}", end="-")


            win_history = deque(maxlen=100)
            win_rate = 0
            model_files = os.listdir(self.exp_dir)
            models_num = len(model_files)
            torch.save(agent.get_policy_net().state_dict(), self.exp_dir / f"model_weights_v{models_num}.pth")
            print(f"----------------Saving model_weights_v{models_num+1}.pth-------------------------")

            new_version = DQN(actions=self.columns).to(device)
            new_version.load_state_dict(agent.get_policy_net().state_dict())
            new_version.eval()

            env.stored_models.append(new_version)


if __name__ == "__main__":
    print(2)
    trainer = DQNTrainer(columns=7, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=50000, replay_size=100000, tau=0.005, lr=5e-4, win_rate=0.60, experiment_name="1-step lookahead")
