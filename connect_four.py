import numpy as np
import torch
import random
from typing import Final
import gymnasium as gym
from enum import Enum

rows: Final = 6
columns: Final = 7

class Turn(Enum):
    AGENT = 1
    OPPONENT = 2

class ConnectFour(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, rows, columns))


    def get_turn(self):
        return self.turn


    def get_terminated(self):
        return self.terminated


    def step(self, action):
        # Agent's turn
        self.turn = Turn.AGENT

        row1 = self.find_slot(action)
        self.insert(row1, action)
        self.print_info(action)
        self.check_game_completion(row1, action)
        if self.terminated:
            return self.board_to_tensor(), self.reward, self.terminated, False, {}

        # Opponent's turn
        self.turn = Turn.OPPONENT

        column = self.opponent_action()
        row2 = self.find_slot(column)
        self.insert(row2, column)
        self.print_info(column)
        self.check_game_completion(row2, column)
        
        self.turn = Turn.AGENT
        return self.board_to_tensor(), self.reward, self.terminated, False, {}


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros([rows, columns])
        self.turn = random.choice(list(Turn))
        if (self.turn == Turn.OPPONENT):
            column = self.opponent_action()
            row = self.find_slot(column)
            self.insert(row, column)
            self.print_info(column)
        self.reward = 0
        self.terminated = False

        self.turn = Turn.AGENT
        return self.board_to_tensor(), {}


    def insert(self, row, column):
        if (self.turn == Turn.AGENT):
            self.board[row][column] = Turn.AGENT.value
        elif (self.turn == Turn.OPPONENT):
            self.board[row][column] = Turn.OPPONENT.value
        else:
            raise Exception("Wrong turn value")


    def find_slot(self, column):
        j = -1
        # from the bottom, up
        for i in range(rows - 1, -1, -1):
            if self.board[i][column] == 0:
                j = i
                break
            else:
                continue

        if j == -1:
            raise Exception(f"Number {column} column is full")

        return j


    def check_game_completion(self, row, column):
        if (self.connect_four(row, column)):
            print(f"{self.turn.name} won!")
            self.terminated = True

            if self.turn == Turn.AGENT:
                self.reward = 1
            else:
                self.reward = -1
        elif not (np.any(self.board[0] == 0)):
            print("It's a Draw, the Board is full!")
            self.terminated = True


    def connect_four(self, row, column):
        directions = []
        directions.append(self.board[:, column]) # vertical
        directions.append(self.board[row, :]) # horizontal
        directions.append(np.diag(self.board, k= column - row)) # positive diagonal
        directions.append(np.diag(np.fliplr(self.board), k= (columns - column -1) - row)) # negative diagonal

        for dir in directions:
            dir = map(int, dir)
            dir_to_str = "".join(map(str, dir))
            # if players disc appears 4 times in a row in a direction
            print(f"{str(self.turn.value)*4} - {dir_to_str}")
            if (str(self.turn.value)*4 in dir_to_str):
                return True
        
        return False


    def opponent_action(self):
        return self.random_action()


    def random_action(self):
        masks = self.get_valid_actions();
        indexes = np.where(masks == 1)

        action = np.random.choice(indexes[0])
        return action



    def board_to_tensor(self):
        agent = (self.board == Turn.AGENT.value).astype(np.float32)
        opponent = (self.board == Turn.OPPONENT.value).astype(np.float32)

        return torch.tensor(np.stack([agent, opponent], axis=0), dtype=torch.float32)


    def get_valid_actions(self):
        return (self.board[0] == 0).astype(int)


    def print_info(self, column):
        print(self.board)
        print(self.turn)
        print(column)


c4 = ConnectFour()
ch = 'y'
while ch != 'n':
    c4.reset()

    while not c4.get_terminated():
        column = int(input("Choose column: "))
        c4.step(column)

    ch = input("Want to continue? (y/n): ")

