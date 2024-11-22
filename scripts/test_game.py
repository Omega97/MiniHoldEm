import numpy as np
from src.game import Game
from src.agents import Agent


def main(n_players=2, n_cards=3, power=2):
    np.random.seed(0)

    game = Game(n_players=n_players,
                n_cards=n_cards,
                power=power,
                random_state=0)

    players = [Agent(game),
               Agent(game),
               ]

    game.play(players, verbose=True)


if __name__ == '__main__':
    main()
