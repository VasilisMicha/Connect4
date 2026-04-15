import torch
import numpy as np
import os
from pathlib import Path
from model import DQN

ROWS = 6
COLUMNS = 7

# Define pieces to match your training environment exactly
AI_PIECE = 1
HUMAN_PIECE = 2

def get_latest_model_path():
    """Finds the model with the highest version number in the models folder."""
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    if not exp_dir.exists():
        print("No models directory found!")
        return None

    experiments = [f for f in exp_dir.iterdir() if f.is_dir()]
    print("!!!!!!!!!!!!!!!!")
    print(experiments)
    if not experiments:
        return None
    latest_experiment = max(experiments, key=lambda f: f.stat().st_ctime)
    print("!!!!!!!!!!!!!!!!")
    print(latest_experiment)
    
    files = [f for f in os.listdir(latest_experiment) if f.endswith('.pth')]
    print("!!!!!!!!!!!!!!!!")
    print(files)
    if not files:
        print("No saved models found!")
        return None
        
    # Sort files by the version number (e.g., model_weights_v23.pth -> 23)
    files.sort(key=lambda x: int(x.split('_v')[1].split('.pth')[0]))
    return latest_experiment / files[-1]

def print_board(board):
    """Prints the board nicely to the console."""
    print("\n  0 1 2 3 4 5 6")
    print(" ---------------")
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLUMNS):
            if board[r][c] == AI_PIECE:
                row_str += " \033[91mX\033[0m" # Red X for AI
            elif board[r][c] == HUMAN_PIECE:
                row_str += " \033[94mO\033[0m" # Blue O for Human
            else:
                row_str += " ."
        print(row_str + " |")
    print(" ---------------\n")

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def check_win(board, piece):
    """Mathematical check for 4-in-a-row (much faster than strings)."""
    # Check horizontal locations
    for c in range(COLUMNS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Check vertical locations
    for c in range(COLUMNS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Check positively sloped diagonals
    for c in range(COLUMNS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Check negatively sloped diagonals
    for c in range(COLUMNS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def board_to_tensor(board, device):
    """Converts the board to the exact 2-channel format the AI expects."""
    agent_channel = (board == AI_PIECE).astype(np.float32)
    opponent_channel = (board == HUMAN_PIECE).astype(np.float32)
    return torch.tensor(np.stack([agent_channel, opponent_channel], axis=0), dtype=torch.float32, device=device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the AI
    model_path = get_latest_model_path()
    print("=========")
    print(model_path)
    if not model_path:
        return
        
    print(f"Loading Final Boss: {model_path.name}")
    ai_net = DQN(actions=COLUMNS).to(device)
    ai_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    ai_net.eval() # Put model in evaluation mode
    
    # 2. Setup the Game
    board = np.zeros((ROWS, COLUMNS))
    game_over = False
    
    # Let the user choose who goes first
    turn_input = input("Do you want to go first? (y/n): ").strip().lower()
    turn = HUMAN_PIECE if turn_input == 'y' else AI_PIECE
    
    print_board(board)
    
    # 3. Game Loop
    while not game_over:
        if turn == HUMAN_PIECE:
            col = -1
            while True:
                try:
                    col = int(input("Make your move (0-6): "))
                    if col >= 0 and col <= 6 and is_valid_location(board, col):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
                    
            row = get_next_open_row(board, col)
            board[row][col] = HUMAN_PIECE
            
            if check_win(board, HUMAN_PIECE):
                print_board(board)
                print("🎉 YOU WIN! The AI still has a lot to learn.")
                game_over = True
                
            turn = AI_PIECE
            
        elif turn == AI_PIECE:
            print("AI is thinking...")
            state_tensor = board_to_tensor(board, device).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                q_values = ai_net(state_tensor)
                
                # Mask out full columns so the AI doesn't try an illegal move
                valid_mask = (board[0] == 0).astype(int)
                mask_torch = torch.from_numpy(valid_mask).bool().to(device)
                masked_values = q_values.masked_fill(~mask_torch, -1e20)
                
                # Get the column with the highest Q-value
                col = masked_values.max(1).indices.item()
                
            row = get_next_open_row(board, col)
            board[row][col] = AI_PIECE
            print(f"AI drops piece in column {col}.")
            
            if check_win(board, AI_PIECE):
                print_board(board)
                print("💀 AI WINS! The Final Boss has defeated you.")
                game_over = True
                
            turn = HUMAN_PIECE
            
        if not game_over:
            print_board(board)
            
        # Check for draw
        if not game_over and not (board[0] == 0).any():
            print("It's a DRAW!")
            game_over = True

if __name__ == "__main__":
    main()
