import numpy as np
from src.game import Game
from src.agents import TrainingAgent
from src.fitness import Fitness


def main(n_players=2, n_cards=3, power=2,
         seed=0, random_state=0):
    np.random.seed(seed)
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)
    agents = [TrainingAgent(game), TrainingAgent(game)]
    fitness = Fitness(len(agents), n_cards, game_tree=TrainingAgent.GAME_TREE)
    v = fitness.compute(agents)
    print(f'\n{np.round(v, 3)}')


if __name__ == '__main__':
    main()
