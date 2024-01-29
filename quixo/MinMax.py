import random
from game import Game, Move, Player
from copy import deepcopy
import numpy as np

class MinMax(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', play_as) -> tuple[tuple[int, int], Move]:
        best_move, _ = self.minimax(game, depth=2, alpha=float('-inf'), beta=float('inf'), maximizing_player=(True if play_as == 'X' else False))
        return best_move

    def minimax(self, game: 'Game', depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple[tuple[int, int], float]:
        #print(f'DEPTH: {depth}')
        
        if depth == 0 or game.check_winner() != -1:
            #print(f'EVAL: {self.evaluate_board(game)}')
            #print(f'BOARD: {game.get_board()}')
            return None, self.evaluate_board(game)

        legal_moves = self.get_legal_moves(game)
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            #print('im here')
            #print(legal_moves)
            #print(game.get_board())
            for move in legal_moves:
                new_game = deepcopy(game)
                new_game._Game__move(move[0], move[1], game.get_current_player())
                _, eval_score = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_game = deepcopy(game)
                new_game._Game__move(move[0], move[1], game.get_current_player())
                _, eval_score = self.minimax(new_game, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def get_legal_moves_1(self, game: 'Game') -> list[tuple[tuple[int, int], Move]]:
        legal_moves = []
    
        for row in range(5):
            for col in range(5):
                actual_state = deepcopy(game)
                acceptable = actual_state._Game__take((row,col), game.get_current_player())
                if acceptable:
                    for slide in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                        new_actual_state = deepcopy(actual_state)
                        acceptable = new_actual_state._Game__slide((row, col), slide)
                        if acceptable:
                            legal_moves.append(((row,col), slide))
        #print(f'LEGAL MOVES: {legal_moves}')
        return legal_moves
    

    def get_legal_moves(self, game:'Game') -> list[tuple[tuple[int, int], Move]]:
        legal_moves = []
        for row in range(5):
            for col in range(5):
                for slide in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                    state = deepcopy(game)
                    acceptable = state._Game__move((row, col), slide, game.get_current_player())
                    if acceptable:
                        legal_moves.append(((row,col), slide))
        return legal_moves



    def evaluate_board(self, game: 'Game') -> float:
        current_player = game.get_current_player()
        opponent_player = 1 - current_player

        # Check if the current player has won
        if game.check_winner() == current_player:
            return 100.0
        # Check if the opponent player has won
        elif game.check_winner() == opponent_player:
            return -100.0
        else:
            # Evaluate based on the difference in the number of pieces
            current_player_pieces = np.count_nonzero(game.get_board() == current_player)
            opponent_player_pieces = np.count_nonzero(game.get_board() == opponent_player)
            piece_difference = current_player_pieces - opponent_player_pieces

            # Evaluate based on the positions of the pieces
            position_score = 0.0
            for row in range(5):
                for col in range(5):
                    if game.get_board()[row, col] == current_player:
                        position_score += self.position_value(row, col)

            # Evaluate based on line completion factor
            line_completion_score = self.line_completion_factor(game, current_player)

            # Combine the factors
            final_score = piece_difference + 0.1 * position_score + 0.2 * line_completion_score

            return final_score

    def line_completion_factor(self, game: 'Game', player_id: int) -> float:
        line_score = 0.0

        # Check rows
        for row in range(5):
            line_score += self.evaluate_line(game.get_board()[row, :], player_id)

        # Check columns
        for col in range(5):
            line_score += self.evaluate_line(game.get_board()[:, col], player_id)

        # Check diagonals
        line_score += self.evaluate_line(np.diag(game.get_board()), player_id)
        line_score += self.evaluate_line(np.diag(np.fliplr(game.get_board())), player_id)

        return line_score

    def evaluate_line(self, line: np.ndarray, player_id: int) -> float:
        # Count the number of player's pieces in the line
        player_pieces = np.count_nonzero(line == player_id)

        # Check if the line has potential for completion
        if player_pieces == 4 and -1 in line:
            return 0.5  # Assign a higher score if completing the line is imminent
        elif player_pieces == 3 and -1 in line:
            return 0.2  # Assign a moderate score for lines with three pieces and one empty slot
        else:
            return 0.0  # No line completion potential


    def evaluate_board_1(self, game: 'Game') -> float:
        current_player = game.get_current_player()
        opponent_player = 1 - current_player

        # Check if the current player has won
        if game.check_winner() == current_player:
            return 100.0
        # Check if the opponent player has won
        elif game.check_winner() == opponent_player:
            return -100.0
        else:
            # Evaluate based on the difference in the number of pieces
            current_player_pieces = np.count_nonzero(game.get_board() == current_player)
            opponent_player_pieces = np.count_nonzero(game.get_board() == opponent_player)
            piece_difference = current_player_pieces - opponent_player_pieces

            # Evaluate based on the positions of the pieces
            position_score = 0.0
            for row in range(5):
                for col in range(5):
                    if game.get_board()[row, col] == current_player:
                        position_score += self.position_value(row, col)

            # Combine the piece difference and position score
            final_score = piece_difference + 0.1 * position_score

            return final_score

    def position_value(self, row: int, col: int) -> float:
        # Assign values to different positions on the board
        position_values = [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 8, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1]
        ]
        return position_values[row][col]