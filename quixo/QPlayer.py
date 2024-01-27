from game import Game, Move, Player, Move
import random
from collections import defaultdict
from tqdm.auto import tqdm
from itertools import product
import numpy as np
from copy import deepcopy

POSITION = [pos for pos in product((0,4), range(5))] + [pos for pos in product((1,2,3), (0,4))]
MOVE = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
ACTION = [a for a in product(POSITION, MOVE)]
COUNTER = 0

print(ACTION[0])

class QPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.env = None

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        #from_pos = (random.rand
        #int(0, 4), random.randint(0, 4))
        #move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        #return from_pos, move
        self.env.current_player = 'X'
        player_id = 0
        state = tuple(self.env.game.get_board().flatten().tolist())
        #self.env.game.print()
        available_moves = self.env.available_moves()
        going_to_win, action = one_move_to_win(self.env, player_id)
        if not going_to_win:
            action = self.agent.choose_action(state, available_moves, play_as = 'X', playing = True)
        else:
            print('-------------------one move for X to win-----------------------')
            print(action)
        return action


class Environment:
    def __init__(self):
        self.game = Game()
        #self.board = self.game._board 
        self.current_player = 'X'  # Player 'X' starts the game

    def print_board(self):
        self.game.print()

    def available_moves(self, board = None):
        player_id = 0 if self.current_player == 'X' else 1
        moves = []

        for a in ACTION:
            if self.game.valid(from_pos=(a[0][0],a[0][1]), slide=a[1], player_id= player_id):
                moves.append(a)
        
        random.shuffle(moves)
        return moves

    def make_move(self, action):
        player_id = 0 if self.current_player == 'X' else 1

        from_pos = (action[0][0], action[0][1])
        slide = action[1]

        self.game.move(from_pos=from_pos, slide= slide, player_id=player_id)

        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        winner_id = self.game.check_winner()

        if winner_id == 0:
            return 'X'
        elif winner_id == 1:
            return 'O'
        else:
            return None

    def game_over(self):
        global COUNTER
        x = self.check_winner()
        y = False

        COUNTER = COUNTER + 1
        if COUNTER == 100:
            y = True
            COUNTER = 0
        #return x is not None or y
        return x is not None
    
    def reset(self):
        self.game._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player = 'X'

# Q-Learning agent to play Tic Tac Toe
class QLearningAgent:
    def __init__(self, epsilon, alpha=0.5, gamma=0.1):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = defaultdict(float)  # Q-table to store state-action values
        self.env = None
        self.usefullness = 0

    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, player):
        if player == 'X':
            old_value = self.get_q_value(state, action)
            l = [self.get_q_value(next_state, a) for a in self.env.available_moves(next_state)]
            if len(l) > 0:
                best_next_action = max([self.get_q_value(next_state, a) for a in self.env.available_moves(next_state)])
            else:
                best_next_action = 0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * best_next_action)
            self.q_table[(state, action)] = new_value
        else:
            old_value = self.get_q_value(state, action)
            l = [self.get_q_value(next_state, a) for a in self.env.available_moves(next_state)]
            if len(l) > 0:
                best_next_action = min([self.get_q_value(next_state, a) for a in self.env.available_moves(next_state)])
            else:
                best_next_action = 0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * best_next_action)
            self.q_table[(state, action)] = new_value


    
    def choose_action(self, state, available_moves, play_as = None, playing = False):
        if not playing: #training
            if random.uniform(0, 1) < self.epsilon.get():
                return random.choice(available_moves)
            else:
                if play_as == 'X' :
                    return max(available_moves, key=lambda a: self.get_q_value(state, a))
                else:
                    return min(available_moves, key=lambda a: self.get_q_value(state, a))

        else:
            if play_as == 'X':
                a = max(available_moves, key=lambda a: self.get_q_value(state, a))
                #print(a)
                if self.get_q_value(state, a) > 0:
                    self.usefullness += 1

                return a
            else:
                return min(available_moves, key=lambda a: self.get_q_value(state, a))
            
class RandomAgent:
    def __init__(self):
        pass

    def choose_action(self, state, available_moves, play_as = None, playing = False):
        return random.choice(available_moves)

def train(episodes):
    epsilon = EpsilonScheduler(low = 0.1, high = 1, num_round = episodes)
    agent = QLearningAgent(epsilon= epsilon)
    env = Environment()
    agent.env = env
    
    for episode in tqdm(range(episodes)):
        env.reset()
        state = tuple(env.game.get_board().flatten().tolist())

        while not env.game_over():
            env.print_board()
            available_moves = env.available_moves()
            player = env.current_player
            player_id = 0 if env.current_player == 'X' else 1
            print(f'current player {player_id}')

            #going_to_win, action = one_move_to_win(env, player_id)
            #if not going_to_win:
            #    action = agent.choose_action(state, available_moves, play_as=player)
            action = agent.choose_action(state, available_moves, play_as=player)


            print(action)

            augmented_states, augmented_actions = generate_augmentation(deepcopy(env.game.get_board()),deepcopy(action))
            #if env.current_player == 'X':

            env.make_move(action)
            next_state = tuple(env.game.get_board().flatten().tolist())
            
            if env.check_winner() == 'X':
                reward = 1

            elif env.check_winner() == 'O':
                reward = -1

            else:
                #reward = intermediate_reward(env, player)
                reward = 0
            
            for state, action in zip(augmented_states, augmented_actions):
                agent.update_q_value(state, action, reward, next_state, player = player)

            state = next_state
        
        agent.epsilon.update()
    
    return agent

def agent_vs_agent(agentX, agentO):
    env = Environment()
    state = tuple(env.board.flatten().tolist())
    
    while not env.game_over():
        if env.current_player == 'X':
            available_moves = env.available_moves()
            action = agentX.choose_action(state, available_moves, play_as = 'X', playing = True)
            env.make_move(action)

        else:
            available_moves = env.available_moves()
            action = agentO.choose_action(state, available_moves, play_as = 'O', playing = True)
            env.make_move(action)

        
        state = tuple(env.board.flatten().tolist())

    
    return env.check_winner()

class EpsilonScheduler():
    def __init__(self, low, high, num_round):
        self.low = low
        self.high = high
        self.num_round = num_round
        self.step = (high - low) / num_round

        self.counter = 0

    def get(self):
        return_val = self.high - self.counter * self.step 
        return return_val if return_val > self.low else self.low
    
    def update(self):
        self.counter += 1
    
def intermediate_reward(env, player):
    game = env.game
    board = game.get_board()
    #print(board)
    p = 0
    #print(player)
    if player == 'X':
        p = 1
    reward = 0
    #print("player: "+str(p))
    count_center = 0
    for i in range(1, 4):
        for j in range(1, 4):
            element = board[i][j]
            #print(element)
            if element == p:
                count_center += 1

    count_good = 0
    for i in range(0, 5):  
        count_on_rows = 0
        count_on_cols = 0
        for j in range(0, 5):
            element_row = board[i][j]
            element_col = board[j][i]
            if element_row == p:
                count_on_rows += 1
            if element_col == p:
                count_on_cols += 1
        if count_on_rows == 4 or count_on_cols == 4:
            count_good += 1
      
    diag = np.diag(board)
    count_diag = 0
    count_good_diag = 0
    for e in diag:
        if e == p:
            count_diag += 1
    if count_diag == 4:
        count_good += 1
        count_good_diag += 1
    
    opposite_diag = [board[i][4 - i] for i in range(5)]
    count_opp_diag = 0
    for e in opposite_diag:
        if e == p:
            count_opp_diag += 1
    if count_opp_diag == 4:
        count_good += 1
        count_good_diag += 1
    
    #print("diagonal: "+str(count_good_diag))
    #print("vertical and horizontal: "+str(count_good-count_good_diag))
    #print("total good: "+str(count_good))
    c0 = 0.3
    c1 = 1 - c0
    reward = c0*(count_center/9) + c1*(count_good/12)
    if player == 'O':
        reward = -reward
    #print(reward)
    return reward

def generate_augmentation(board, action):
    augmented_states = []
    augmented_actions= []

    for i in range(0, 4):
        rotated_board = np.rot90(board, k = i, axes=(0,1))
        flipped_board = np.flip(rotated_board, axis= 0)

        rotated_state = tuple(rotated_board.flatten().tolist())
        flipped_state = tuple(flipped_board.flatten().tolist())

        from_pos = (action[0][0], action[0][1])
        slide = action[1]

        rotated_pos, flipped_pos = rotate_and_flip(from_pos, i)
        rotated_slide = Move._value2member_map_[(slide.value + 1) % 4]

        if slide == Move.LEFT or slide == Move.RIGHT:
            flipped_slide = slide
        elif slide == Move.TOP:
            flipped_slide = Move.BOTTOM
        else:
            flipped_slide = Move.TOP

        rotated_action = ((rotated_pos[0],rotated_pos[1]), rotated_slide)
        flipped_action = ((flipped_pos[0],flipped_pos[1]), flipped_slide)
        
        augmented_states.append(rotated_state)
        augmented_states.append(flipped_state)
        augmented_actions.append(rotated_action)
        augmented_actions.append(flipped_action)

    return augmented_states, augmented_actions
        
def rotate_and_flip(pos, i):
    x = pos[0]-2
    y = pos[1]-2

    rotated_xy = None
    flipped_xy = None

    if i == 0:
        rotated_xy = (x,y)
        flipped_xy = (x, -y)
    elif i == 1:
        rotated_xy = (-y, x)
        flipped_xy = (-y, -x)
    elif i == 2:
        rotated_xy = (-x,-y)
        flipped_xy = (-x, y)
    elif i == 3:
        rotated_xy = (y, -x)
        flipped_xy = (y, x)
    
    rotated_pos = (rotated_xy[0] + 2, rotated_xy[1] + 2)
    flipped_pos = (flipped_xy[0] + 2, flipped_xy[1] + 2)

    return rotated_pos, flipped_pos

def change(a):
    x = np.zeros(a.shape, dtype= int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0:
                x[i][j] = 1
            elif a[i][j] == 1:
                x[i][j] = 0
            else:
                x[i][j] = a[i][j]
    
    return x


def one_move_to_win(env, player_id):
    board = env.game.get_board()

    #vertical
    for col in range(5):
        if np.sum(board[:, col] == player_id) == 4:
            row = np.argwhere(board[:,col] != player_id).ravel()[0]
            possible, action =  horizontal_slide(row, col, board, player_id)
            if possible:
                return possible, action  
    #horizontal
    for row in range(5):
        if np.sum(board[row, :] == player_id) == 4:
            col = np.argwhere(board[row,:] != player_id).ravel()[0]
            possible, action =  vertical_slide(row, col, board, player_id)
            if possible:
                return possible, action

            
    #Diagonal
    if np.sum(np.diag(board) == player_id) == 4:
        diag_pos = np.argwhere(np.diag(board) != player_id).ravel()[0]
        possible, action = horizontal_slide(diag_pos, diag_pos, board, player_id)
        if possible:
            return possible, action
        possible, action = vertical_slide(diag_pos, diag_pos, board, player_id)
        if possible:
            return possible, action
        
    if np.sum(np.diag(np.flip(board, axis=1)) == player_id) == 4:
        diag_pos = np.argwhere(np.diag(np.flip(board, axis=1)) != player_id).ravel()[0]
        possible, action = horizontal_slide(diag_pos, 4 - diag_pos, board, player_id)
        if possible:
            return possible, action
        possible, action = vertical_slide(diag_pos, 4 - diag_pos, board, player_id)
        if possible:
            return possible, action
        
    return False, None
        
def horizontal_slide(row, col, board, player_id):
    if col == 0:
        if row == 0 or row == 4:
            selected = np.argwhere(np.logical_or(board[row,:] == player_id, board[row,:] == -1)).ravel()
            selected = selected[selected > col][0]

            return True, ((selected, row), Move.LEFT)
        else:
            if board[row][4] == player_id or board[row][4] == -1:
                return True, ((4,row), Move.LEFT)

    if col == 4:
        if row == 0 or row == 4:
            selected = np.argwhere(np.logical_or(board[row,:] == player_id, board[row,:] == -1)).ravel()
            selected = selected[selected < col][0]

            return True, ((selected, row), Move.RIGHT)
        else:
            if board[row][0] == player_id or board[row][0] == -1:
                return True, ((0,row), Move.RIGHT)

    if col > 0:
        if board[row, col-1] == player_id and np.any(np.logical_or(board[row,col:] == player_id, board[row,col:] == -1)):
            selected = np.argwhere(np.logical_or(board[row,:] == player_id, board[row,:] == -1)).ravel()
            selected = selected[selected >= col][0]

            if row != 0 and row != 4:
                if selected == 4:
                    return True, ((selected, row), Move.LEFT)
            else:
                    return True, ((selected, row), Move.LEFT)

    if col < 4:
        if board[row, col+1] == player_id and np.any(np.logical_or(board[row,:col] == player_id, board[row,:col] == -1)):
            selected = np.argwhere(np.logical_or(board[row,:] == player_id, board[row,:] == -1)).ravel()
            selected = selected[selected <= col][0]

            if row != 0  and row != 4:
                if selected == 0:
                        return True, ((selected,row), Move.RIGHT)
            else:
                    return True, ((selected,row), Move.RIGHT)
    
    return False, None

def vertical_slide(row, col, board, player_id):
    if row == 0:
        if col == 0 or col == 4:
            selected = np.argwhere(np.logical_or(board[:,col] == player_id, board[:,col] == -1)).ravel()
            selected = selected[selected > row][0]

            return True, ((col, selected), Move.TOP)
        else:
            if board[4][col] == player_id or board[4][col] == -1:
                return True, ((col,4), Move.TOP)

    if row == 4:
        if col == 0 or col == 4:
            selected = np.argwhere(np.logical_or(board[:,col] == player_id, board[:,col] == -1)).ravel()
            selected = selected[selected < row][0]

            return True, ((col, selected), Move.BOTTOM)
        else:
            if board[0][col] == player_id or board[0][col] == -1:
                return True, ((col,0), Move.BOTTOM)
            
    if row > 0:
        if board[row - 1, col] == player_id and np.any(np.logical_or(board[row:,col] == player_id, board[row:,col] == -1)):
            selected = np.argwhere(np.logical_or(board[:,col] == player_id, board[:,col] == -1)).ravel()
            selected = selected[selected >= row][0]

            if col != 0 and col != 4:
                if selected == 4:
                    return True, ((col,selected), Move.TOP)
            else:
                    return True, ((col,selected), Move.TOP)
    if row < 4:
        if board[row + 1, col] == player_id and np.any(np.logical_or(board[:row,col] == player_id, board[:row,col] == -1)):
            selected = np.argwhere(np.logical_or(board[:,col] == player_id, board[:,col] == -1)).ravel()
            selected = selected[selected <= row][0]

            if col != 0  and col != 4:
                if selected == 0:
                    return True, ((col,selected), Move.BOTTOM)
            else:
                    return True, ((col, selected), Move.BOTTOM)
    return False, None
    


if __name__ == '__main__':
    for i in range(0,4):
        print(rotate_and_flip((3,0),i))
 
         
