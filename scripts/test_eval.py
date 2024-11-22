from src.fitness import Fitness
from src.game import Game
from src.training import TrainingAgent


def main(n_players=2, n_cards=6, power=2,
        n_repeat=100, n_games=1000, random_state=0):
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)
    agents = list()

    path = f'..\\data\\best_agent_{n_players}_{n_cards}_{power} (2).pkl'
    agents.append(TrainingAgent(game).load(path))
    agents.append(TrainingAgent(game, sigma=0.))

    assert len(agents) == n_players

    game.play(agents, n_games=5, verbose=True)

    # theoretical fitness
    fitness = Fitness(len(agents), n_cards, game_tree=TrainingAgent.GAME_TREE)
    v = fitness.compute(agents)
    print(f'\ntheoretical fitness = {v[0]:+.2f}')

    # measured fitness
    game.reset_fitness()
    for _ in range(3):
        game.play(agents, n_games=n_games)
        print(f'   measured fitness = {game.get_fitness()[0]:.2f} ({game.hands_count} hands)')


if __name__ == '__main__':
    main()
