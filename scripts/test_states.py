import numpy as np
from src.game import Game
from src.agents import Agent
import pickle


class TestAgent(Agent):
    STATES = set()

    def __init__(self, game: Game):
        super().__init__(game)

    def __call__(self, state):
        """0=fold, 1=call, 2=raise"""

        t = tuple(state["actions"])
        if t not in TestAgent.STATES:
            TestAgent.STATES.add(t)

        p = np.array([1., 1., 4.])
        p *= state['legal_actions']
        p /= p.sum()
        return np.random.choice(len(p), p=p)


def main(n_players=2, n_cards=6, power=7, n_games=1_000):
    np.random.seed(0)

    game = Game(n_players=n_players,
                n_cards=n_cards,
                power=power,
                random_state=0)

    players = [TestAgent(game),
               TestAgent(game),
               ]

    game.play(players, n_games=n_games)

    for t in sorted(TestAgent.STATES):
        print(t)
    with open(fr'tree_{n_players}_{power}.pkl', 'wb') as f:
        t = tuple(sorted(TestAgent.STATES))
        pickle.dump(t, f)

    print(len(TestAgent.STATES))


if __name__ == '__main__':
    main()
