from collections import namedtuple, deque
import torch.optim as optim
from model import DQN
import numpy as np
import torch
import math
import random
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, state_size, action_size, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, replay_size, tau, lr, env, device):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(replay_size)
        self.replay_size = replay_size
        self.tau = tau
        self.lr = lr
        self.policy_net = DQN(actions=self.action_size).to(device)
        self.target_net = DQN(actions=self.action_size).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.steps_done = 0
        self.device = device
        self.env = env


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                values = self.policy_net(state)
                mask_torch = torch.from_numpy(self.env.get_valid_actions()).bool().to(self.device)
                masked_values = values.masked_fill(~mask_torch, -1e20)
                return masked_values.max(1).indices.view(1, 1)
        else:
            valid_actions = np.where(self.env.get_valid_actions() == 1)[0]
            random_action = random.choice(valid_actions)
            return torch.tensor([[random_action]])


    def store_transition(self, state, action, stats):
        next_state = stats[0].unsqueeze(0)
        reward = torch.tensor([[stats[1]]])
        terminated = stats[2]
        if terminated:
            self.memory.push(state, action, None, reward)
        else:
            self.memory.push(state, action, next_state, reward)
        
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros((self.batch_size, 1), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values.unsqueeze(1)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def update_target_network(self):
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        # Soft Update: It uses your TAU ($\tau$) value to slowly merge the policy_net weights into the target_net.Stability: This ensures the target values don't jump around too much, which prevents the training from becoming chaotic.
