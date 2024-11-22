"""
Fitness = sum(leaf i, value_i * leaf_proba_i)
leaf_proba_i = prod(branch node j, proba_j)
proba_i = exp(x_i) / sum(node actions k, exp(x_k))

dproba_i/dx_j = proba_i * (delta(i, j) - proba_j)
dlog(leaf_proba_i)/dx_j = sum(branch_i node k, delta(j, k)) = 1
dlp_i/dx_j = 1
lp_i = log(leaf_proba_i)
dleaf_proba_i/dx_j = dlp_i/dx_j * leaf_proba_i
dFitness/dx_j = sum(leaf i, value_i * dleaf_proba_i/dx_j)
              = sum(leaf i, value_i * leaf_proba_i * dlp_i/dx_j)
              = sum(leaf i, value_i * leaf_proba_i)
dFitness/dx_j = sum(leaf l, Fitness_l * 1(x_j is in branch of leaf i)
"""
import numpy as np
from src.game import Game
from src.agents import TrainingAgent
from src.fitness import Fitness


def main(seed=0, random_state=0,
         n_players=2, n_cards=3, power=2):
    np.random.seed(seed)
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)
    agents = [TrainingAgent(game), TrainingAgent(game)]
    fitness = Fitness(len(agents), n_cards, game_tree=TrainingAgent.GAME_TREE)
    v = fitness.compute(agents)
    print(f'\n{np.round(v, 3)}')


if __name__ == '__main__':
    main()
