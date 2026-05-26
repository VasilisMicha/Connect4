import csv
from pathlib import Path
from collections import deque

project_root = Path(__file__).resolve().parent.parent

class Logger:
    def __init__(self, exp_dir):
        self.logs_file = exp_dir / "training_stats.csv"
        
        self.headers = [
            "Games Played", 
            "Win Rate against Models", 
            "Draw Rate against Models", 
            "Win Rate against Random Moves", 
            "Average Game Length", 
            "Average Loss", 
            "Average Max Q value"
        ]
        
        with open(self.logs_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

        self.games_played = 0
        self.win_history_against_models = deque(maxlen=100)
        self.win_rate_against_models = 0
        self.win_history_against_random = deque(maxlen=100)
        self.win_rate_against_random = 0
        self.draw_history_against_models = deque(maxlen=100)
        self.draw_rate_against_models = 0
        self.interval_game_length = 0
        self.interval_loss = 0
        self.interval_max_q = 0

    def log_training(self, reward, opponent_is_model, steps, avg_loss, avg_max_q_value):
        self.games_played += 1
        self.interval_loss += avg_loss
        self.interval_max_q += avg_max_q_value
        self.interval_game_length += steps

        if opponent_is_model:
            if reward == 1:
                self.win_history_against_models.append(1)
                self.draw_history_against_models.append(0)
            elif reward == -1:
                self.win_history_against_models.append(0)
                self.draw_history_against_models.append(0)
            else:
                self.win_history_against_models.append(0)
                self.draw_history_against_models.append(1)
            
            self.win_rate_against_models = sum(self.win_history_against_models) / len(self.win_history_against_models)
            self.draw_rate_against_models = sum(self.draw_history_against_models) / len(self.draw_history_against_models)

        else:
            self.win_history_against_random.append(1 if reward == 1 else 0)
            self.win_rate_against_random = sum(self.win_history_against_random) / len(self.win_history_against_random)

        if self.games_played % 100 == 0:
            log_data = {
                "Games Played": self.games_played,
                "Win Rate against Models": self.win_rate_against_models,
                "Draw Rate against Models": self.draw_rate_against_models,
                "Win Rate against Random Moves": self.win_rate_against_random,
                "Average Game Length": self.interval_game_length / 100,
                "Average Loss": self.interval_loss / 100,
                "Average Max Q value": self.interval_max_q / 100
            }
            
            with open(self.logs_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(log_data)
            
            self.interval_game_length = 0
            self.interval_loss = 0
            self.interval_max_q = 0
