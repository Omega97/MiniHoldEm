from src.game import Game
from src.training import TrainingAgent
from src.agents import AGENTS_DIR


def main(n_players=3, n_cards=6, power=4,
         random_state=0, n_games=20):
    game = Game(n_players=n_players, n_cards=n_cards, power=power, random_state=random_state)

    path = AGENTS_DIR + f'best_agent_{n_players}_{n_cards}_{power}.pkl'
    agents = [TrainingAgent(game).load(path) for _ in range(n_players)]

    assert len(agents) == n_players

    # example games
    game.play(agents, n_games=n_games, verbose=True)


if __name__ == '__main__':
    main()
