import numpy as np
import pickle
from src.game import Game
from src.agents import ConstantAgent


class GameTree(Game):
    TREE = dict()

    def __init__(self, n_players, n_cards, power, do_move_dealer=True, random_state=None):
        super().__init__(n_players, n_cards, power,
                         do_move_dealer=do_move_dealer,
                         random_state=random_state)
        self._parent_hash = None

    def add_node(self, state, parent_hash=None):
        """Add node to game tree"""
        h = state['decision_hash']
        if h not in self.TREE:
            info = {'bets': state['bets'],
                    'depth': 0}
            if state['is_game_over']:
                info['is_game_over'] = True
            if np.any(state['folded']):
                info['folded'] = state['folded']
            if parent_hash is not None:
                info['parent'] = parent_hash
                info['depth'] = self.TREE[parent_hash]['depth'] + 1
            self.TREE[h] = info

    def reset_game(self):
        super().reset_game()
        self.add_node(self.get_state())

    def _get_player_action(self) -> int:
        action = super()._get_player_action()
        state = self.get_state()
        self.parent_hash = state['decision_hash']
        return action

    def _execute_player_action(self, action):
        super()._execute_player_action(action)
        self.add_node(self.get_state(), parent_hash=self.parent_hash)


def test(n_games=200, n_players=2, n_cards=3, power=2):
    game = GameTree(n_players=n_players, n_cards=n_cards, power=power,
                    do_move_dealer=False, random_state=0)

    p = np.array([1., 1., 4.])
    agents = [ConstantAgent(game, p=p) for _ in range(2)]
    game.play(agents, n_games=n_games)

    print()
    for k, v in game.TREE.items():
        depth = v["depth"]
        c = " *"[v.get("is_game_over", 0)]
        print(f'{k} {c}{"  " * depth}{v}')

    print(f'\n{len(game.TREE)} nodes')

    file_name = f'game_tree_{n_players}_{n_cards}_{power}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(game.TREE, f)
    print(f'\n>>> Saved to {file_name}')


if __name__ == '__main__':
    test()
