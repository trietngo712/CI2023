import random
from game import Game, Move, Player
from QPlayer import QPlayer, Environment, train


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':

    player1 = QPlayer()
    player1.agent = train(1000)
    counter = 0

    for _ in range(100):
        g = Game()
        g.print()

        env = Environment()
        env.game = g
        player1.env = env

        player2 = RandomPlayer()
        winner = g.play(player2, player1)
        if winner == 0:
            counter+= 1
        g.print()

    print(f"Winner: Player {winner}")
    print(counter)
