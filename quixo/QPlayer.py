from game import Game, Move, Player, Move
import random
from collections import defaultdict
from tqdm.auto import tqdm
from itertools import product
import numpy as np

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
        state = tuple(self.env.board.flatten().tolist())
        available_moves = self.env.available_moves()
        return self.agent.choose_action(state, available_moves, play_as = 'X', playing = False)


class Environment:
    def __init__(self):
        self.game = Game()
        self.board = self.game._board 
        self.current_player = 'X'  # Player 'X' starts the game

    def print_board(self):
        self.game.print()

    def available_moves(self, board = None):
        player_id = 0 if self.current_player == 'X' else 1
        moves = []

        for a in ACTION:
            if self.game.valid(from_pos=(a[0][0],a[0][1]), slide=a[1], player_id= player_id):
                moves.append(a)
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
        if COUNTER == 1000:
            y = True
            COUNTER = 0
        return x is not None or y
    
    def reset(self):
        self.game._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.board = self.game._board
        self.current_player = 'X'

# Q-Learning agent to play Tic Tac Toe
class QLearningAgent:
    def __init__(self, epsilon, alpha=0.5, gamma=0.1):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = defaultdict(float)  # Q-table to store state-action values
        self.env = None
    
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
                return max(available_moves, key=lambda a: self.get_q_value(state, a))
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
        state = tuple(env.board.flatten().tolist())
        while not env.game_over():
            available_moves = env.available_moves()
            player = env.current_player
            action = agent.choose_action(state, available_moves, play_as=player)
            env.make_move(action)
            next_state = tuple(env.board.flatten().tolist())
            
            if env.check_winner() == 'X':
                reward = 1


            elif env.check_winner() == 'O':
                reward = -1

            else:
                reward = 0
            
            
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
        self.num_round = num_round * 9
        self.step = (high - low) / num_round

        self.counter = 0

    def get(self):
        return_val = self.high - self.counter * self.step 
        return return_val
    
    def update(self):
        self.counter += 1
    


