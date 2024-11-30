import numpy as np
from src.game import Game
from copy import deepcopy
import pickle

GAME_TREE_DIR = '../data/trees/'
AGENTS_DIR = '../data/agents/'


def save_game_tree(tree, n_players, power):
    file_name = GAME_TREE_DIR + fr'game_tree_{n_players}_{power}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(tree, f)


def load_game_tree(n_players, power):
    file_name = GAME_TREE_DIR + fr'game_tree_{n_players}_{power}.pkl'
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class Agent:
    def __init__(self, game: Game):
        self.game = game
        self.n_players = game.n_players
        self.n_cards = game.n_cards
        self.n_chips = game.n_chips

    def start_game(self, state):
        """start-of-game procedures"""
        pass

    def __call__(self, state, verbose=False):
        """0=fold, 1=call, 2=raise"""
        p = np.ones(len(state['legal_actions']))
        p *= state['legal_actions']
        p /= p.sum()
        action = np.random.choice(len(p), p=p)
        return action

    def end_game(self, reward, global_state):
        """end-of-game procedures"""
        pass


class ConstantAgent(Agent):
    def __init__(self, game: Game, p=np.ones(3)):
        super().__init__(game)
        self.p = p

    def __call__(self, state, verbose=False):
        """0=fold, 1=call, 2=raise"""
        p = self.p * state['legal_actions']
        p /= p.sum()
        return np.random.choice(len(p), p=p)


class HumanAgent(Agent):
    ACTIONS = np.array(['fold', 'call', 'raise'])
    ACTIONS_MAP = {'1': 1, '2': 2, '3': 0, ' ': 0, '0': 0,
                   'c': 1, 'r': 2, 'f': 0, }
    ACTION_EMOJIS = ['🤚', '📞', '🔼']

    def _print_bets(self, state):

        last_action = [" "] * self.n_players
        for p, a in state.get('actions', []):
            last_action[p] = HumanAgent.ACTION_EMOJIS[a]

        # board
        print('\nBets / Stacks')
        for i in range(self.n_players):
            print(f'Player {i+1})  {state["bets"][i]:3}  / {state["stacks"][i]:3}🪙  {last_action[i]}')

    def _print_gains(self, state):
        print('\nGains')
        for i in range(self.n_players):
            gain = state['stacks'][i] - self.game.n_chips
            if gain != 0:
                print(f'Player {i+1})  {gain:+6.0f}🟡')
            else:
                print(f'Player {i+1}) ')

    def _print_actions(self, state):

        if 'to_call' in state:
            print(f'\n  Pot: {state["pot"]} 🟡   ({state["to_call"]}🪙 to call)')

        # hole
        if 'hole' in state:
            print(f'\n Hole: {self.game.print_card(state["hole"])}')

        # actions
        actions = state.get('actions')
        actors = state.get('actors')
        print()
        if actions is not None:
            print(f'Actions')
            for p, a in zip(actions, actors):
                print(f'Player{p+1:2}) {HumanAgent.ACTION_EMOJIS[a]}')

    @staticmethod
    def _print_legal_actions(state):
        legal_action_set = np.where(state['legal_actions'])[0]
        print()
        for i in range(len(HumanAgent.ACTIONS)):
            if i in legal_action_set:
                print(f' {i} = {HumanAgent.ACTIONS[i]}', end=' ')
        print('\n')

    @staticmethod
    def _get_action(state):
        legal_action_set = np.where(state['legal_actions'])[0]
        action = None

        while True:
            s = input('Action: ')
            if s == '':
                continue
            action = HumanAgent.ACTIONS_MAP.get(s[0])
            if action in legal_action_set:
                break

        return action

    def __call__(self, state, verbose=False):
        """0=fold, 1=call, 2=raise"""
        self._print_bets(state)
        self._print_actions(state)
        self._print_legal_actions(state)
        action = self._get_action(state)
        return action

    def end_game(self, reward, global_state):
        self._print_bets(global_state)
        self._print_actions(global_state)
        print(f'\n\n>>> Game over 🏁')

        # showdown
        print('\nShowdown')
        for i in range(self.n_players):
            print(f'Player {i+1})  {self.game.print_card(global_state["hands"][i])}')

        print(f'\nWinner: {global_state["winner"]+1} 🏆')

        self._print_gains(global_state)

        print('\n' + '=' * 40)
        input()


def get_tree(n_players=2, power=7, root=r'..\data'):
    path = fr'{root}\tree_{n_players}_{power}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def softmax(logits):
    v = np.exp(logits)
    return v / np.sum(v)


class DummyAgent(Agent):

    def __init__(self, game: Game, strategt_size, logits=None, scale=0.1):
        super().__init__(game)
        self.game = game
        self.logits = logits
        if self.logits is None:
            self.logits = np.random.normal(0, scale, (strategt_size, 3))

        self.dealer = None
        self.hole = None
        self.actions = None
        self.winner = None
        self.hands = None
        self.player_id = None
        self.strategy = None

        self.n_players = self.game.n_players
        self.n_cards = self.game.n_cards
        self.power = self.game.power

        self._init_strategy()

    def _init_strategy(self):
        tree = get_tree(self.n_players, self.power)
        self.strategy = dict()
        i = 0
        for p in range(self.n_players):
            for c in range(self.n_cards):
                for t in tree:
                    key = (p, c, t)
                    self.strategy[key] = self.logits[i]
                    i += 1

    def get_proba(self, player: int, card: int, actions):
        """player, card, actions"""
        key = (int(player), int(card), tuple(actions))
        return softmax(self.strategy[key])

    def start_game(self, state):
        self.dealer = state['dealer']
        self.hole = state['hole']
        self.player_id = state['player_id']

    def __call__(self, state, verbose=False):
        """0=fold, 1=call, 2=raise"""
        self.actions = state['actions']

        p = self.get_proba(self.player_id, self.hole, self.actions)
        p *= state['legal_actions']
        p /= p.sum()
        return np.random.choice(len(p), p=p)

    def end_game(self, reward, global_state):
        self.winner = global_state['winner']
        self.actions = global_state['actions']
        self.hands = global_state['hands']


class TrainingAgent(Agent):
    GAME_TREE = None
    n_nodes = None
    internal_node_hashes = None
    leaf_hashes = None

    def __init__(self, game: Game, sigma=0.):
        super().__init__(game)
        self.logits = None
        self.n_players = self.game.n_players
        self.n_cards = self.game.n_cards
        self.power = self.game.power
        self.n_actions = 3

        self._player_id = None
        self._hole = None
        self._dealer = None
        self._actions = None
        self._node_hash = None

        if TrainingAgent.GAME_TREE is None:
            self._load_game_tree()
        self._init_logits(sigma)

    def _load_game_tree(self):
        TrainingAgent.GAME_TREE = load_game_tree(self.n_players, self.power)

        # assign hashes and number of nodes
        TrainingAgent.internal_node_hashes = []
        TrainingAgent.leaf_hashes = []
        for k, v in TrainingAgent.GAME_TREE.items():
            if 'children' not in v:
                TrainingAgent.leaf_hashes.append(k)
            else:
                TrainingAgent.internal_node_hashes.append(k)

        TrainingAgent.leaf_hashes = tuple(TrainingAgent.leaf_hashes)
        TrainingAgent.internal_node_hashes = tuple(TrainingAgent.internal_node_hashes)
        TrainingAgent.n_nodes = len(TrainingAgent.internal_node_hashes)

    def _init_logits(self, sigma):
        """
        The logits are a tensor of shape (n_cards, n_nodes, n_actions),
        where n_nodes is the number of decision-points in the game tree.
        """
        shape = (self.n_cards, TrainingAgent.n_nodes, self.n_actions)
        self.logits = np.random.normal(0, sigma, shape)

    def start_game(self, state):
        self._dealer = state['dealer']
        self._hole = state['hole']
        self._player_id = state['player_id']

    def get_proba(self, hole, node_hash):
        """player, card, actions"""
        n = TrainingAgent.internal_node_hashes.index(node_hash)
        x = self.logits[hole, n, :]
        p = softmax(x)
        return p

    def __call__(self, state, verbose=False, length=24):
        """0=fold, 1=call, 2=raise"""
        self._actions = state['actions']
        self._node_hash = state['decision_hash']
        p = self.get_proba(self._hole, self._node_hash)
        p *= state['legal_actions']
        p /= p.sum()
        if verbose:
            for i in range(3):
                c = self.game.ACTION_EMOJIS[i]
                if state['legal_actions'][i]:
                    bar = '=' * round(p[i] * length)
                    print(f'{c} | {p[i]:7.2%} |{bar}')
                else:
                    print(f'{c} | {"":7} |')
        return np.random.choice(len(p), p=p)

    def save(self, file_name):
        print(f'Saving agent {file_name}')
        with open(file_name, 'wb') as f:
            pickle.dump(self.logits, f)

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self.logits = pickle.load(f)
        return self


def test_1():
    np.random.seed(0)
    game = Game(n_players=2, n_cards=6, power=7)
    strategy_size = ...
    agent = DummyAgent(game, strategy_size)
    for k in agent.strategy:
        print(f'{np.round(softmax(agent.strategy[k]), 2)}  {k}')
    print(len(agent.strategy))


def test_2(n_agents=25, n_games=100, random_state=0, n_epochs=200, n_replace=3):
    np.random.seed(0)

    # environment
    game = Game(n_players=2, n_cards=6, power=7, random_state=random_state)
    strategy_size = ...

    # initialization
    agents = [DummyAgent(game, strategy_size) for _ in range(n_agents)]

    for epoch in range(n_epochs):
        print(epoch)

        # fitness
        scores = np.zeros(n_agents)
        for i in range(n_agents):
            for j in range(i):
                game.play([agents[i], agents[j]], n_games=n_games)
                reward = game.total_rewards / n_games * game.n_chips
                scores[i] += reward[0]
                scores[j] += reward[1]
        scores /= (n_agents-1)

        # sort players by score
        order = np.argsort(scores)
        agents = [agents[i] for i in order]

        for j in range(n_replace):
            # replacement
            agents[j] = deepcopy(agents[-j])

            # mutation
            agents[j].logits += np.random.normal(0, 0.1, agents[j].logits.shape)

    # evaluation
    best_agent = agents[-1]
    # print(np.round(best_agent.logits, 2))
    random_agent = Agent(game)
    game.play([best_agent, random_agent], n_games=200)
    print(np.round(game.total_rewards / game.hands_count * game.n_chips, 4))


if __name__ == '__main__':
    test_1()
    # test_2()
