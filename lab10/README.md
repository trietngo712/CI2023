# Lab 10 of Computational Intelligence

For this lab I collaborated with Enrico Capuano (s317038)

The class TicTacToe defines the functioning of the game itself, giving a good print of the playing table.
Moreover, it has methods to define a move and to check when the game is over and who is the winner. 

In the context of Reinforcement Learning, we implememted a Q-Learning agent, based on a Q table that gathers the Q values for all the moves that an agent may do.
The Q Agent is trained to choose the best move, according to this table. One Q agent will have positive reward, while the other will have negative: according to this, the first one will choose the move with the highest Q value, the other the one with the lowest.

For each agent, the winning rate is computed as the number of winning games over the number of total games.

Moreover, we implemented the method "play_vs_agent" to play TicTacToe ourselves against an agent (that could be the trained one or the random one).