from src.agents import TrainingAgent
import numpy as np
from itertools import permutations


def inverse_permutation_numpy(perm):
    """ Calculates the inverse of a permutation using NumPy. """
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return inv_perm


class Fitness:
    def __init__(self, n_players: int, n_cards: int, game_tree: dict):
        self.n_players = n_players
        self.game_tree = game_tree
        self.n_cards = n_cards

        self.agents = None
        self.orders = [np.roll(np.arange(self.n_players), i) for i in range(n_players)]
        self.inv_orders = [inverse_permutation_numpy(v) for v in self.orders]
        self.n_orders = len(self.orders)
        self.hands = tuple(permutations(range(self.n_cards), self.n_players))
        self.n_hands = len(self.hands)
        self.leaf_hashes = None
        self.n_leaf_hashes = None
        self._compute_leaf_hashes()

        self.shape = (self.n_players, self.n_hands, self.n_leaf_hashes, self.n_players)
        self.proba = None
        self.rewards = None
        self._compute_rewards()

    def get_order(self, i: int) -> np.array:
        return self.orders[i]

    def get_ordered(self, i: int) -> np.array:
        return self.inv_orders[i]

    def _compute_leaf_hashes(self):
        """Compute which nodes of the game tree are leafs"""
        self.leaf_hashes = []
        for k, v in TrainingAgent.GAME_TREE.items():
            if 'children' not in v:
                self.leaf_hashes.append(k)
        self.leaf_hashes = tuple(self.leaf_hashes)
        self.n_leaf_hashes = len(self.leaf_hashes)
        assert self.n_leaf_hashes

    def _get_reward(self, leaf_hash, i_order: int, holes):
        """Compute the reward for each player"""
        leaf = self.game_tree[leaf_hash]
        folded = np.array(leaf.get('folded', np.zeros(self.n_players)), dtype=bool)
        scores = (np.array(holes) + 1) * (1 - folded)
        winner = np.argmax(scores)

        reward = -np.array(leaf['bets'])
        reward[winner] += leaf['pot']

        order = self.get_order(i_order)
        reward = reward[order]
        return reward

    def _compute_rewards(self):
        """ Scores
        * for each position i_order ...
        * for each combination of cards i_h ...
        * for each leaf node i_l ...
        * for each player i ...
        compute the player's score, based on which player
        of those that didn't fold has the strongest hand.
        """
        self.rewards = np.ones(self.shape)
        for i_order in range(len((self.orders))):
            for i_hand, holes in enumerate(self.hands):
                for i_leaf, leaf_hash in enumerate(self.leaf_hashes):
                    reward = self._get_reward(leaf_hash, i_order, holes)
                    self.rewards[i_order, i_hand, i_leaf, :] = reward

    def _compute_branch(self, leaf_hash):
        """
        Get the parents, positions, actions that lead to the leaf
        (equal to the number of internal nodes, root included)
        (game-tree-specific)
        parents: all nodes except for the leaf
        positions: which position made the move in each node (always starts with 0, 1, ..., n, ...)
        actions: which action was made in that node
        """
        leaf = self.game_tree[leaf_hash]
        parents = leaf['branch'][:-1]
        positions = tuple(leaf['positions'][:len(parents)])
        actions = leaf['actions']
        return parents, positions, actions

    def _get_proba(self, i_order, holes, parents, positions, actions):
        """Multiply the probabilities of the right agent making the right move"""
        probability = 1.
        for i_n, (node, position) in enumerate(zip(parents, positions)):
            order = self.get_order(i_order)
            player = self.agents[order[position]]
            parent = self.game_tree[node]
            mask = np.array([i in parent['legal_actions'] for i in range(3)])
            hole = holes[position]
            rho = player.get_proba(hole, node_hash=node)
            rho *= mask
            rho /= rho.sum()
            probability *= rho[actions[i_n]]
        return probability

    def _compute_proba(self):
        """ Probabilities
        * for each position i_p ...
        * for each combination of cards i_h ...
        * for each leaf node i_l ...
        compute the probability of getting to that leaf
        node as a product of the probabilities of the corresponding
        agents making the moves that lead to that leaf node.
        """
        self.proba = np.ones(self.shape)
        for i_order in range(len(self.orders)):
            for i_hand, holes in enumerate(self.hands):
                for i_l, leaf_hash in enumerate(self.leaf_hashes):
                    # build the branch from the leaf node to the root
                    parents, positions, actions = self._compute_branch(leaf_hash)

                    # compute the probability of reaching the leaf node
                    p = self._get_proba(i_order, holes, parents, positions, actions)
                    self.proba[i_order, i_hand, i_l, :] *= p

    def compute(self, agents):
        """
        The fitness is the expected value of each agent's reward,
        calculated by computing the probability of ending in each
        possible leaf node of the game tree, and averaged over all
        the combinations of hands.
        """
        self.agents = agents
        self._compute_proba()
        p_tot = np.sum(self.proba, axis=(0, 1, 2))
        assert np.allclose(p_tot, self.n_hands * self.n_players), f'{p_tot} are not all close to 1.'

        fitness = np.sum(self.proba * self.rewards, axis=(0, 1, 2,))
        fitness /= self.n_hands * self.n_players
        return fitness

    def __call__(self, agents):
        return self.compute(agents)

    def gradient(self):
        """Compute the gradient of the fitness
        relative to the parameters of all the agents.

        F = sum(i_leaf, p_i_leaf * s_i_leaf)
        L = sum(i_leaf, log(p_i_leaf) * s_i_leaf)
          = sum(i_leaf, sum(i_node, log(p_(i_leaf, i_node))) * s_i_leaf)
          = sum((i_leaf, i_node), lp_(i_leaf, i_node) * s_i_leaf)

        dL/dlp_(j,k) = sum((i_leaf, i_node), 1(i_leaf=j, i_node=k) * s_i_leaf)

        """
        ...
