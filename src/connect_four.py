import numpy as np
import os
from pathlib import Path
import torch
import random
from typing import Final
import gymnasium as gym
from enum import Enum
from model import DQN

rows: Final = 6
columns: Final = 7

class Turn(Enum):
    AGENT = 1
    OPPONENT = 2


class ConnectFour(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, rows, columns))
        self.stored_models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.load_models()
        self.opponent = None


    def step(self, action):
        self.reward = 0.0

        # Agent's turn
        self.turn = Turn.AGENT

        row1 = self.find_slot(action)
        self.insert(row1, action)
        self.check_game_completion(row1, action)
        if self.terminated:
            return self.board_to_tensor(), self.reward, self.terminated, False, {}
        elif self.check_horizontal_block(row1, action):
            self.reward += 0.2
        elif self.check_preventative_block(row1, action):
            self.reward += 0.1

        # Opponent's turn
        self.turn = Turn.OPPONENT

        column = self.opponent_action(self.board_to_tensor().unsqueeze(0).to(self.device))
        row2 = self.find_slot(column)
        self.insert(row2, column)
        self.check_game_completion(row2, column)
        
        self.turn = Turn.AGENT
        return self.board_to_tensor(), self.reward, self.terminated, False, {}


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros([rows, columns])
        self.opponent = self.pick_opponent() 
        self.turn = random.choice(list(Turn))
        if (self.turn == Turn.OPPONENT):
            column = self.opponent_action(self.board_to_tensor().unsqueeze(0).to(self.device))
            row = self.find_slot(column)
            self.insert(row, column)
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
            self.terminated = True
            self.reward = 1 if self.turn == Turn.AGENT else -1
        elif not (np.any(self.board[0] == 0)):
            self.terminated = True
            self.reward = 0.2


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
            if (str(self.turn.value)*4 in dir_to_str):
                return True
        
        return False

    
    def check_horizontal_block(self, row, col):
        piece = Turn.AGENT.value
        opp = Turn.OPPONENT.value
        
        current_row = self.board[row, :]
        
        for i in range(max(0, col-3), min(columns-3, col+1)):
            window = list(current_row[i:i+4])
            
            # Does this window have exactly 3 opponent pieces and 1 agent piece?
            if window.count(opp) == 3 and window.count(piece) == 1:
                return True
        return False


    def check_preventative_block(self, row, col):
        piece = Turn.AGENT.value
        opp = Turn.OPPONENT.value
        current_row = self.board[row, :]

        # Sliding 5-slot window
        for i in range(max(0, col-4), min(columns-4, col+1)):
            window = list(current_row[i:i+5])
            
            if window.count(opp) == 2 and window.count(piece) == 1 and window.count(0) == 2:
                return True
        return False



    def get_immediate_threat(self, piece_value):
        valid_actions = np.where(self.get_valid_actions() == 1)[0]
    
        for col in valid_actions:
            row = self.find_slot(col)
        
            self.board[row][col] = piece_value
        
            # Temporarily change the turn so connect_four() checks the right pieces
            original_turn = self.turn
            self.turn = Turn.AGENT if piece_value == Turn.AGENT.value else Turn.OPPONENT
        
            is_win = self.connect_four(row, col)
        
            self.board[row][col] = 0
            self.turn = original_turn
        
            if is_win:
                return col
            
        return None


    def opponent_action(self, state):
        if self.opponent:
            win_move = self.get_immediate_threat(Turn.OPPONENT.value)
            if win_move is not None:
                return win_move
        
            block_move = self.get_immediate_threat(Turn.AGENT.value)
            if block_move is not None:
                return block_move

            flipped_state = torch.flip(state, [1])
            with torch.no_grad():
                values = self.opponent(flipped_state)
                mask_torch = torch.from_numpy(self.get_valid_actions()).bool().to(self.device)
                masked_values = values.masked_fill(~mask_torch, -1e20)
                action = masked_values.max(1).indices.view(1, 1)
                return action.item()
        else:
            return self.random_action()


    def random_action(self):
        masks = self.get_valid_actions();
        indexes = np.where(masks == 1)

        action = np.random.choice(indexes[0])
        return action



    def board_to_tensor(self):
        agent = (self.board == Turn.AGENT.value).astype(np.float32)
        opponent = (self.board == Turn.OPPONENT.value).astype(np.float32)

        return torch.tensor(np.stack([agent, opponent], axis=0), dtype=torch.float32, device=self.device)


    def get_valid_actions(self):
        mask = (self.board[0] == 0).astype(int)
        return np.array(mask)


    def load_models(self):
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        if not models_dir.exists():
            models_dir.mkdir(parents=True)

        model_files = os.listdir(models_dir)
        model_files.sort()
        for model_file in model_files:
            model = DQN(actions=7).to(self.device) 
            model.load_state_dict(torch.load(models_dir / model_file, map_location=self.device, weights_only=True))
            print(f"loaded {model_file}")
            model.eval()
            self.stored_models.append(model)
            
            
    def pick_opponent(self):
        if len(self.stored_models) > 50:
            self.stored_models.pop(0) # save RAM
        elif len(self.stored_models) == 0:
            return None

        if self.stored_models:
            if random.random() < 0.5 or len(self.stored_models) <= 1:
                return self.stored_models[-1] # choose the best model
            else:
                if len(self.stored_models) > 5:
                    start_idx = len(self.stored_models) - 3
                    return self.stored_models[random.randrange(start_idx, len(self.stored_models) - 1)]
                elif len(self.stored_models) > 20:
                    start_idx = len(self.stored_models) - 5
                    return self.stored_models[random.randrange(start_idx, len(self.stored_models) - 1)]
                elif len(self.stored_models) > 40:
                    start_idx = len(self.stored_models) - 20
                    return self.stored_models[random.randrange(start_idx, len(self.stored_models) - 1)]
                else:
                    return self.stored_models[random.randrange(len(self.stored_models) - 1)]

        else:
            return None


    def get_turn(self):
        return self.turn


    def get_terminated(self):
        return self.terminated


    def get_opponent(self):
        return self.opponent

