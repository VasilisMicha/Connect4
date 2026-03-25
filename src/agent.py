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

    def __init__(self, action_size, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, replay_size, tau, lr, env, device):
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
            return torch.tensor([[random_action]], device=self.device)


    def store_transition(self, state, action, stats):
        next_state = stats[0].unsqueeze(0).to(self.device)
        reward = torch.tensor([[stats[1]]], device=self.device)
        terminated = stats[2]
        if terminated:
            self.memory.push(state, action, None, reward)
        else:
            self.memory.push(state, action, next_state, reward)
        
        
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None, None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions)) # converts the batch to a single Transition that contains 4 lists

        # removed finished games
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        # from tuple of tensors, to a single tensor
        state_batch = torch.cat(batch.state) 
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # What the agent guessed the move was worth 
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        avg_max_q = state_action_values.mean().item()
        next_state_values = torch.zeros((self.batch_size, 1), device=self.device)
        with torch.no_grad():
            next_qs = self.target_net(non_final_next_states)
            
            # The top row is index 0. If both channels (agent=0, opponent=1) 
            # are 0 at the top row, the column is valid.
            # Shape of non_final_next_states: (batch, channels, rows, cols)
            valid_mask = (non_final_next_states[:, 0, 0, :] == 0) & (non_final_next_states[:, 1, 0, :] == 0)
            
            masked_next_qs = next_qs.masked_fill(~valid_mask, -1e20)
            next_state_values[non_final_mask] = masked_next_qs.max(1).values.unsqueeze(1)

        # What the move was actually worth (Reward + Future).
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item(), avg_max_q


    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            with torch.no_grad():
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def get_policy_net(self):
        return self.policy_net
