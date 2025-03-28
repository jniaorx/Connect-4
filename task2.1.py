import numpy as np
import game_utils
from typing import Tuple
from connect_four import ConnectFour
import time

# constants
MOVE_NONE = -1
MAX_DEPTH = 4
## Task 2.1: Defeat the Baby Agent
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
        self.transposition_table = {}
        self.move_times = []
        pass
    def make_move(self, state):
        """
        Determines and returns the next move for the agent based on the current game state.
        Parameters:
        -----------
        state : np.ndarray
            A 2D numpy array representing the current, read-only state of the game board. 
            The board contains:
            - 0 for an empty cell,
            - 1 for Player 1's piece,
            - 2 for Player 2's piece.
        Returns:
        --------
        int
            The valid action, ie. a valid column index (col_id) where this agent chooses to drop its piece.
        """
        """ YOUR CODE HERE """
        start_time = time.time()

        game = ConnectFour()
        game.state = state.copy()
        # check available row for a particular col
        def get_available_row(col_id: int) -> int:
            for row in range(game.size()[0] - 1, -1, -1):
                if game.state[row, col_id] == 0:
                    return row
            # return -1 if the col is already full
            return -1
        # check if the next move gives an immediate win
        def is_winning_move(col_id: int, player_id: int) -> bool:
            row = get_available_row(col_id)
            if row == -1:
                return False
            # create a copy of the game
            copy_game = game.state.copy()
            # perform the move
            game_utils.step(copy_game, col_id, player_id, in_place=True)
            # return if the move is winning move or not
            return game_utils.is_win(copy_game)
        # check for moves that lead to opponent's win the next move
        def leads_to_opponent_win(game: ConnectFour, col_id: int, player_id: int) -> bool:
            # get the row where the piece would be placed
            row = get_available_row(col_id)
            if row == -1:
                return False
            
            # create a copy of the game state and make the move
            next_state = game.state.copy()
            game_utils.step(next_state, col_id, player_id, in_place=True)
            # check if opponent can win in any column after this move
            opponent_id = 3 - player_id
            for opp_col in range(game.size()[1]):
                # get available row for opponent's potential move
                opp_row = get_available_row(opp_col)
                if opp_row == -1:
                    continue  # skip if column is full
                # try opponent's move
                next_state[opp_row, opp_col] = opponent_id
                # check if this creates a win
                if game_utils.is_win(next_state):
                    return True
                # undo the test move
                next_state[opp_row, opp_col] = 0
            return False
        # evaluation function to check the goodness of a particular state
        def evaluation_func(game: ConnectFour) -> int:
            # init score and total moves made
            score = 0
            # count number of moves made so far
            total_moves_made = np.sum(game.state != 0)
            # apply scoring logics:
            # 1. check for immediate winning moves for both current player and opponent player
            for col in game.get_valid_col_id():
                if is_winning_move(col, self.player_id):
                    return 10000 - total_moves_made
                if is_winning_move(col, 3 - self.player_id):
                    return -10000  # very high negative score to force blocking
                if leads_to_opponent_win(game, col, self.player_id):
                    return -8000  # heavy penalty for moves that let opponent win next turn
                
            # 2. center col control (it is more strategic to place discs in center column)
            centre_col = game.size()[1] // 2 # calculate the index of centre col
            # count the number of disc put in the centre col by the current player
            centre_col_count = np.sum(game.state[:, centre_col] == self.player_id) 
            # adjust score according centre_col_count (the higher the better)
            score += centre_col_count * 3
            
            for row in range(game.size()[0]):
                for col in range(game.size()[1]):
                    # 3. Strategic positional scoring
                    # check vertical to see if there are two consecutive pieces of current player
                    if col + 1 < game.size()[1] and game.state[row, col + 1] == self.player_id:
                        # reward if there are two consecutive pieces
                        score += 10
                    if game.state[row, col] == self.player_id:
                        # check horizontal
                        if row + 1 < game.size()[0] and game.state[row + 1, col] == self.player_id:
                            score += 10                        
                        # check diagonal slope down left
                        if row + 1 < game.size()[0] and col + 1 < game.size()[1] and game.state[row + 1, col + 1] == self.player_id:
                            score += 10
                        # check diagonal slope down right
                        if row + 1 < game.size()[0] and col - 1 >= 0 and game.state[row + 1, col - 1] == self.player_id:
                            score += 10
                    
                    # 4. Perform blocking moves
                    # check potential win for opponent (check if there are three consecutive discs)
                    if game.state[row, col] == 3 - self.player_id:
                        # check vertical
                        if col + 1 < game.size()[1] and game.state[row, col + 1] == 3 - self.player_id:
                            if col + 2 < game.size()[1] and game.state[row, col + 2] == 0:
                                # penalize the score heavily if the opponent is close to winning (to block the move)
                                score -= 1000
                        # check horizontal
                        if row + 1 < game.size()[0] and game.state[row + 1, col] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and game.state[row + 2, col] == 0:
                                score -= 1000
                        # check diagonal slope down left
                        if row + 1 < game.size()[0] and col + 1 < game.size()[1] and game.state[row + 1, col + 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col + 2 < game.size()[1] and game.state[row + 2, col + 2] == 0:
                                score -= 1000
                        # check diagonal slope down right
                        if row + 1 < game.size()[0] and col - 1 >= 0 and game.state[row + 1, col - 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col - 2 >= 0 and game.state[row + 2, col - 2] == 0:
                                score -= 1000
                        # check vertical
                        if col + 1 < game.size()[1] and game.state[row, col + 1] == 3 - self.player_id:
                            if col + 2 < game.size()[1] and game.state[row, col + 2] == 3 - self.player_id:
                                # penalize the score heavily if the opponent is close to winning (to block the move)
                                score -= 2000
                        # check horizontal
                        if row + 1 < game.size()[0] and game.state[row + 1, col] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and game.state[row + 2, col] == 3 - self.player_id:
                                score -= 2000
                        # check diagonal slope down left
                        if row + 1 < game.size()[0] and col + 1 < game.size()[1] and game.state[row + 1, col + 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col + 2 < game.size()[1] and game.state[row + 2, col + 2] == 3 - self.player_id:
                                score -= 2000
                        # check diagonal slope down right
                        if row + 1 < game.size()[0] and col - 1 >= 0 and game.state[row + 1, col - 1] == 3 - self.player_id:
                            if row + 2 < game.size()[0] and col - 2 >= 0 and game.state[row + 2, col - 2] == 3 - self.player_id:
                                score -= 2000
            return score
        
        # perform negamax function with alpha-beta pruning
        def negamax(game: ConnectFour, player_id: int, depth: int, alpha: int, beta: int) -> Tuple[int, int]:
            #create a unique key for the position that includes depth
            state_hash = (game.state.tobytes(), depth, player_id)
            
            # lookup position in transposition table
            if state_hash in self.transposition_table:
                stored_score, stored_move, stored_depth = self.transposition_table[state_hash]
                # only use the stored result if it was searched to at least the same depth
                if stored_depth >= depth:
                    return stored_score, stored_move
            
            if depth == 0 or game.is_end():
                return evaluation_func(game), MOVE_NONE
            
            # init best score and best move
            best_score = float('-inf')
            best_move = MOVE_NONE
            
            for col in game.get_valid_col_id():
                row = get_available_row(col)
                if row == -1:
                    continue
                next_game = ConnectFour()
                next_game.state = game.state.copy()
                next_game.step((col, player_id)) # current player's turn
                # perform negamax recursively to find the best score
                score, _ = negamax(next_game, 3 - player_id, depth - 1, -beta, -alpha)
                score = -score # negate score for other player
                # update best_score if the current score if higher than the best score
                if score > best_score:
                    best_score = score
                    best_move = col
                
                # update alpha
                alpha = max(alpha, best_score)
                # do alpha-beta pruning
                if alpha >= beta:
                    break
            
            # store the result in transposition table
            self.transposition_table[state_hash] = (best_score, best_move, depth)
            return best_score, best_move
        # first check if we have a winning move
        for col in game.get_valid_col_id():
            if is_winning_move(col, self.player_id):
                return col
        # then check if opponent has a winning move and block it
        opponent_id = 3 - self.player_id
        for col in game.get_valid_col_id():
            if is_winning_move(col, opponent_id):
                return col
        
        # obtain next best move from negamax function
        # _, best_move = negamax(game, self.player_id, MAX_DEPTH, float('-inf'), float('inf'))
        # start_time = time.time()
        for depth in range(1, MAX_DEPTH + 1):
            _, best_move = negamax(game, self.player_id, depth, float('-inf'), float('inf'))
            if time.time() - start_time > 0.3:  # check time after each move
                break  # stop further evaluations once the time exceeds 1 second
        end_time = time.time()
        move_time = end_time - start_time
        self.move_times.append(move_time)
        return best_move
        """ YOUR CODE END HERE """


def test_task_2_1():
    assert(True)
    # Upload your code to Coursemology to test it against our agent.
    print("All test cases passed!")

if __name__ == "__main__":
    from simulator import GameController, HumanAgent
    from connect_four import ConnectFour
    import numpy as np
    import game_utils
    import time

    # Initialize the game board
    board = ConnectFour()

    # Initialize the agents
    human_agent = HumanAgent(1)
    ai_agent = AIAgent(2)

    # Set up the game with one human and one AI agent
    game = GameController(board=board, agents=[human_agent, ai_agent])

    # Start timing the game run
    start_time = time.time()

    # Run the game
    game.run()

    # End timing the game run
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print move times for each agent
    print(f"AI Agent move times: {ai_agent.move_times}")
    print(f"Average move time for AI Agent: {np.mean(ai_agent.move_times)} seconds")
    print(f"Human Agent move times: {elapsed_time}")

