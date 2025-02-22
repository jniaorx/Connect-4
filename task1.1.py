import numpy as np
import game_utils
from typing import Tuple
from connect_four import ConnectFour

# Declare constants
MOVE_NONE = -1
MAX_DEPTH = 4

class AIAgent(object):
    """
    A class representing an agent that plays Connect Four.
    """
    def __init__(self, player_id=1):
        """Initializes the agent with the specified player ID.

        Parameters:
        -----------
        player_id : int
            The ID of the player assigned to this agent (1 or 2).
        """
        self.player_id = player_id

    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.
        """
        game = ConnectFour()
        game.state = state.copy()

        def get_row_count(col_id: int) -> int:
            """Get the next available row index for a piece to be placed in the specified column."""
            for row in range(game.size()[0] - 1, -1, -1):
                if game.state[row, col_id] == 0:
                    return row
            return -1  # Return -1 if the column is full

        def is_winning_move(col_id: int, player_id: int) -> bool:
            """Determine if the next move gives an immediate win."""
            temp_game = game.state.copy()
            game_utils.step(temp_game, col_id, player_id, in_place=True)
            return game_utils.is_win(temp_game)

        def evaluation_func(game: ConnectFour) -> int:
            score = 0
            total_moves = np.sum(game.state != 0)  # Count total moves made

            # Center column control
            centre_col = game.size()[1] // 2
            centre_count = np.sum(game.state[:, centre_col] == self.player_id)
            score += centre_count * 3

            # Check for immediate winning moves and blocking moves
            for player in [self.player_id, 3 - self.player_id]:
                for move in game.get_valid_col_id():
                    if is_winning_move(move, player):
                        if player == self.player_id:
                            return 1000 - total_moves  # Winning move for AI
                        else:
                            score -= 1000  # Block opponent's winning move

            # Additional scoring for strategic positioning (horizontal, vertical, diagonal)
            for row in range(game.size()[0]):
                for col in range(game.size()[1]):
                    if game.state[row, col] == self.player_id:
                        # Check horizontal
                        if col + 1 < game.size()[1] and game.state[row, col + 1] == self.player_id:
                            score += 10
                        # Check vertical
                        if row + 1 < game.size()[0] and game.state[row + 1, col] == self.player_id:
                            score += 10
                        # Check diagonal /
                        if row + 1 < game.size()[0] and col + 1 < game.size()[1] and game.state[row + 1, col + 1] == self.player_id:
                            score += 10
                        # Check diagonal \
                        if row + 1 < game.size()[0] and col - 1 >= 0 and game.state[row + 1, col - 1] == self.player_id:
                            score += 10

                    # Check for potential horizontal win for opponent
                    if game.state[row, col] == 3 - self.player_id:
                        # Check for potential vertical win for opponent
                        if row + 1 < game.size()[0] and game.state[row + 1, col] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and game.state[row + 2, col] == 3 - self.player_id:
                                # Opponent has three in a row vertically, apply penalty
                                score -= 100  # Block opponent's vertical winning move
                                
                        # Check if the opponent is one move away from winning horizontally
                        if col + 1 < game.size()[1] and game.state[row, col + 1] == 3 - self.player_id:
                            if col + 2 < game.size()[1] and game.state[row, col + 2] == 3 - self.player_id:
                                # Opponent has three in a row horizontally, apply penalty
                                score -= 100  # Block opponent's horizontal winning move

                        
                        # Check for potential diagonal winning moves for opponent (slope down to the right)
                        if row + 1 < game.size()[0] and col + 1 < game.size()[1] and game.state[row + 1, col + 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col + 2 < game.size()[1] and game.state[row + 2, col + 2] == 3 - self.player_id:
                                # Opponent has three in a row diagonally (slope down to the right), apply penalty
                                score -= 100  # Block opponent's diagonal winning move (slope down to the right)

                        # Check for potential diagonal winning moves for opponent (slope down to the left)
                        if row + 1 < game.size()[0] and col - 1 >= 0 and game.state[row + 1, col - 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col - 2 >= 0 and game.state[row + 2, col - 2] == 3 - self.player_id:
                                # Opponent has three in a row diagonally (slope down to the left), apply penalty
                                score -= 100  # Block opponent's diagonal winning move (slope down to the left)

            return score


        def negamax(game: ConnectFour, depth: int, player_id: int, alpha: int, beta: int) -> Tuple[int, int]:
            """Perform negamax function with alpha-beta pruning."""
            if depth == 0 or game.is_end():
                return evaluation_func(game), MOVE_NONE

            best_score = float('-inf')
            best_move = MOVE_NONE

            for col in game.get_valid_col_id():
                # Use get_row_count to find the next available row
                row = get_row_count(col)
                if row == -1:  # If the column is full, skip it
                    continue

                next_game = ConnectFour()
                next_game.state = game.state.copy()
                next_game.step((col, player_id))  # Current player's turn

                score, _ = negamax(next_game, depth - 1, 3 - player_id, -beta, -alpha)
                score = -score  # Negate score for the other player

                if score > best_score:
                    best_score = score
                    best_move = col

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break

            return best_score, best_move

        # Check for immediate winning moves first
        for move in game.get_valid_col_id():
            row = get_row_count(move)
            if row != -1:  # Proceed only if the move is valid
                if is_winning_move(move, self.player_id):
                    return move  # Return winning move immediately

        _, best_move = negamax(game, MAX_DEPTH, self.player_id, float('-inf'), float('inf'))
        return best_move

def test_task_1_1():
    from utils import check_step, actions_to_board
    
    # Test case 1
    res1 = check_step(ConnectFour(), 1, AIAgent)
    assert(res1 == "Pass")
    
    # Test case 2
    res2 = check_step(actions_to_board([0, 0, 0, 0, 0, 0]), 1, AIAgent)
    assert(res2 == "Pass")
    
    # Test case 3
    res2 = check_step(actions_to_board([4, 3, 4, 5, 5, 1, 4, 4, 5, 5]), 1, AIAgent)
    assert(res2 == "Pass")
    
    print("All testcases passed!")

# Run the test function
if __name__ == "__main__":
    test_task_1_1()