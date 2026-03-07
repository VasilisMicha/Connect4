import numpy as np
import random
from typing import Final
import gymnasium as gym
from enum import Enum

rows: Final = 6
columns: Final = 7

class Turn(Enum):
    PLAYER1 = 1
    PLAYER2 = -1

class ConnectFour(gym.Env):
    def __init__(self):
        self.board = np.zeros([rows, columns], dtype=np.int_)
        self.turn = random.choice(list(Turn))
        self.reward = 0
        self.terminated = False
        self.action_space = gym.spaces.Discrete(columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, rows, columns))


    def get_terminated(self):
        return self.terminated


    def step(self, action):
        column = action
        row = self.find_slot(column)
        self.insert(row, column)
        print(self.board)
        print(self.turn)
        
        # terminate if the board is full or a connect four was achieved
        if (self.connect_four(row, column)):
            print(f"{self.turn.name} won!")
            self.terminated = True
            if self.turn == Turn.PLAYER1:
                self.reward = 1
            else:
                self.reward = -1
        elif not (np.any(self.board[0] == 0)):
            print("Board is full!")
            self.terminated = True


        if self.turn == Turn.PLAYER1:
            self.turn = Turn.PLAYER2
        elif self.turn == Turn.PLAYER2:
            self.turn = Turn.PLAYER1

        return self.board_to_tensor()


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros([rows, columns], dtype=np.int_)
        self.turn = random.choice(list(Turn))
        self.reward = 0
        self.terminated = False

        return self.board_to_tensor(), {}


    def insert(self, row, column):
        if (self.turn == Turn.PLAYER1):
            self.board[row][column] = Turn.PLAYER1.value
        elif (self.turn == Turn.PLAYER2):
            self.board[row][column] = Turn.PLAYER2.value
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


    def connect_four(self, row, column):
        directions = []
        directions.append(self.board[:, column]) # vertical
        directions.append(self.board[row, :]) # horizontal
        directions.append(np.diag(self.board, k= column - row)) # positive diagonal
        directions.append(np.diag(np.fliplr(self.board), k= (columns - column -1) - row)) # negative diagonal

        for dir in directions:
            dir_to_str = "".join(map(str, dir))
            # if players disc appears 4 times in a row in a direction
            if (str(self.turn.value)*4 in dir_to_str):
                return True
        
        return False


    def board_to_tensor(self):
        player = (self.board == self.turn.value).astype(int)
        opponent = (self.board == -self.turn.value).astype(int)

        value = 1.0 if self.turn == Turn.PLAYER1 else 0.0
        turn_indicator = np.full((rows, columns), value, dtype=np.float32)

        return np.stack([player, opponent, turn_indicator], axis=0)


    def get_valid_actions(self):
        return (self.board[0] == 0).astype(int)


    def print_board(self):
        print(self.board)


b = ConnectFour()
ch = 'y'
while ch != 'n':
    while not b.get_terminated():
        column = int(input("Choose column: "))
        b.step(column)


    ch = input("Want to continue? (y/n): ")

