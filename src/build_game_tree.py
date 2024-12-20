import numpy as np
from src.game import Game
from src.agents import ConstantAgent, save_game_tree, load_game_tree


class GameTree(Game):
    TREE = dict()

    def __init__(self, n_players, n_cards, power=7,
                 do_move_dealer=False, random_state=None):
        super().__init__(n_players, n_cards, power,
                         do_move_dealer=do_move_dealer, random_state=random_state)
        self.parent_hash = None

    def load_game_tree(self):
        try:
            GameTree.TREE = load_game_tree(self.n_players, self.power)
        except FileNotFoundError:
            print(f'No game tree found')

    def add_node(self, state, parent_hash=None):
        """Add node to game tree"""
        h = state['decision_hash']
        if h not in self.TREE:
            if parent_hash is None:
                branch = [h]
            else:
                branch = self.TREE[parent_hash]['branch'] + [h]
            if parent_hash is None:
                positions = [state['position']]
            else:
                positions = self.TREE[parent_hash]['positions'] + [state['position']]
            info = {'bets': state['bets'],
                    'pot': state['pot'],
                    'depth': 0,
                    'position': state['position'],
                    'legal_actions': set(),
                    'branch': branch,
                    'positions': positions,
                    'actions': state['actions'],
                    }
            if np.any(state['folded']):
                info['folded'] = state['folded']
            if parent_hash is not None:
                info['parent'] = parent_hash
                info['depth'] = self.TREE[parent_hash]['depth'] + 1
            self.TREE[h] = info
        self.parent_hash = h

    def reset_game(self):
        super().reset_game()
        # self.add_node(self.get_state())

    def _get_player_action(self) -> int:
        action = super()._get_player_action()
        return action

    def assign_reward(self):
        self.add_node(self.get_state(), parent_hash=self.parent_hash)

    def _execute_player_action(self, action):
        self.add_node(self.get_state(), parent_hash=self.parent_hash)
        super()._execute_player_action(action)

        # add child to parent
        parent = self.TREE[self.parent_hash]
        h = self.get_state()['decision_hash']
        if 'children' not in parent:
            parent['children'] = dict()
        parent['children'][h] = action
        parent['legal_actions'].add(action)


def print_game_tree(tree: dict):
    print()
    for k, v in tree.items():
        depth = v["depth"]
        c = " *"[int('children' not in v)]
        print(f'{k} {c}{"  " * depth}{v}')


def test(n_players=4, power=4, n_games=300):
    game = GameTree(n_players=n_players, n_cards=n_players+1, power=power,
                    do_move_dealer=False, random_state=None)

    game.load_game_tree()

    p = np.array([1., 2., 4.])
    agents = [ConstantAgent(game, p=p) for _ in range(n_players)]

    while True:
        # play games
        game.play(agents, n_games=n_games)

        # show results
        # print_game_tree(game.TREE)
        print(f'{len(game.TREE)} nodes')

        # save game tree
        save_game_tree(game.TREE, n_players, power)

    # print(f'\n>>> Saved game tree')


if __name__ == '__main__':
    test()
